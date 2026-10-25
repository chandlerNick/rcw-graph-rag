import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import lancedb
import networkx as nx
import numpy as np


DEFAULT_EMBEDDER = "st:CSI-lab/Washington-state-law-embedding-model-Base"


def hash_embedder(dim=512):
    def emb(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            tokens = words + [a + "_" + b for a, b in zip(words, words[1:])]
            for token in tokens:
                hashed = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little")
                out[i, hashed % dim] += 1 if (hashed >> 63) else -1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    return emb


def get_embedder(spec=DEFAULT_EMBEDDER):
    if spec == "hash":
        return hash_embedder()
    if spec.startswith("st:"):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(spec[3:])
        return lambda texts: model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
    raise ValueError(f"unknown embedder {spec!r}; use 'hash' or 'st:<model>'")


def build(data_dir, db_dir, embed_spec=DEFAULT_EMBEDDER, batch=512):
    data_path, output_path = Path(data_dir), Path(db_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def read_jsonl(filename):
        with (data_path / filename).open(encoding="utf8") as source:
            for line in source:
                yield json.loads(line)

    graph = nx.MultiDiGraph()
    with (data_path / "titles.json").open(encoding="utf8") as source:
        for title in json.load(source):
            graph.add_node(
                f"title:{title['title_number']}",
                kind="Title",
                name=title["title_name"],
                url=title["url"],
            )
    with (data_path / "chapters.json").open(encoding="utf8") as source:
        for chapter in json.load(source):
            graph.add_node(
                f"chapter:{chapter['chapter_number']}",
                kind="Chapter",
                name=chapter["chapter_name"],
                url=chapter["url"],
            )

    embed = get_embedder(embed_spec)
    database = lancedb.connect(str(output_path / "lancedb"))
    rows, table, section_count = [], None, 0

    def flush():
        nonlocal rows, table
        if not rows:
            return

        vectors = embed([row["embedding_text"] for row in rows])
        for row, vector in zip(rows, vectors):
            row["vector"] = np.asarray(vector, dtype=np.float32).tolist()
        if table is None:
            table = database.create_table("sections", rows, mode="overwrite")
        else:
            table.add(rows)
        rows = []

    fields = (
        "id",
        "section_title",
        "text",
        "embedding_text",
        "chapter_number",
        "chapter_name",
        "title_number",
        "title_name",
        "url",
        "last_amended_year",
        "effective_note",
    )
    for section in read_jsonl("sections.jsonl"):
        graph.add_node(
            section["id"],
            kind="Section",
            name=section["section_title"],
            url=section["url"],
            chapter=section["chapter_number"],
            title=section["title_number"],
            last_amended=section["last_amended_year"],
        )
        rows.append({key: section[key] for key in fields})
        section_count += 1
        if len(rows) >= batch:
            flush()
    flush()

    for edge in read_jsonl("edges.jsonl"):
        if edge["type"] == "AMENDED_BY" and edge["target"] not in graph:
            graph.add_node(edge["target"], kind="Law")
        if (
            edge.get("resolved") is False
            or edge["source"] not in graph
            or edge["target"] not in graph
        ):
            continue
        key = (
            edge["type"]
            if edge["type"] != "AMENDED_BY"
            else f"AMENDED_BY:{edge.get('cite', '')}"
        )
        graph.add_edge(edge["source"], edge["target"], key=key, rel=edge["type"])

    with (output_path / "graph.pkl").open("wb") as output:
        pickle.dump(graph, output, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "embedder": embed_spec,
        "sections": section_count,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }
    (output_path / "meta.json").write_text(json.dumps(metadata), encoding="utf8")
    print(
        f"built {section_count} sections, graph {graph.number_of_nodes()} nodes / "
        f"{graph.number_of_edges()} edges, embedder={embed_spec} -> {output_path}"
    )


class RCWGraphRAG:
    def __init__(self, db_dir):
        output_path = Path(db_dir)
        with (output_path / "graph.pkl").open("rb") as source:
            self.G = pickle.load(source)
        with (output_path / "meta.json").open(encoding="utf8") as source:
            self.meta = json.load(source)
        self.tbl = lancedb.connect(str(output_path / "lancedb")).open_table("sections")
        self.emb = get_embedder(self.meta["embedder"])

    def vector_search(self, query, k=5, where=None):
        prefixed_query = (
            f"Represent this sentence for searching relevant passages: {query}"
        )
        query_vector = np.asarray(self.emb([prefixed_query])[0], dtype=np.float32)
        query_builder = self.tbl.search(query_vector).metric("cosine")
        if where:
            query_builder = query_builder.where(where)
        return query_builder.limit(k).to_list()

    def related(self, section_id, hops=1, rels=("REFERENCES", "NEXT_SECTION")):
        seen, frontier = {section_id: 0}, {section_id}
        for hop in range(1, hops + 1):
            next_frontier = set()
            for node in frontier:
                edges = list(self.G.out_edges(node, data=True)) + list(
                    self.G.in_edges(node, data=True)
                )
                for source, target, data in edges:
                    neighbor = target if source == node else source
                    if (
                        data["rel"] in rels
                        and self.G.nodes[neighbor]["kind"] == "Section"
                        and neighbor not in seen
                    ):
                        seen[neighbor] = hop
                        next_frontier.add(neighbor)
            frontier = next_frontier
        seen.pop(section_id)
        return seen

    def get(self, ids):
        ids = list(ids)
        if not ids:
            return []
        query = ",".join("'" + section_id.replace("'", "''") + "'" for section_id in ids)
        records = {
            row["id"]: row
            for row in self.tbl.search()
            .where(f"id IN ({query})")
            .limit(len(ids))
            .to_list()
        }
        return [records[section_id] for section_id in ids if section_id in records]

    def retrieve(self, query, k=5, hops=1, max_extra=10, where=None):
        hits = self.vector_search(query, k, where)
        hit_ids = [hit["id"] for hit in hits]
        extra = {}
        for section_id in hit_ids:
            for related_id, hop_count in self.related(section_id, hops).items():
                if related_id not in hit_ids and related_id not in extra:
                    extra[related_id] = (hop_count, section_id)
        extra = dict(
            sorted(extra.items(), key=lambda item: item[1][0])[:max_extra]
        )
        graph_context = [
            {
                **record,
                "via": extra[record["id"]][1],
                "hops": extra[record["id"]][0],
            }
            for record in self.get(extra)
        ]
        return {"hits": hits, "graph_context": graph_context}

    def context(self, query, **kwargs):
        result = self.retrieve(query, **kwargs)

        def format_record(record, label):
            return (
                f"[{record['id']}] {record['section_title']} "
                f"({label}; Title {record['title_number']}, "
                f"ch. {record['chapter_number']})\n{record['text']}"
            )

        return "\n\n".join(
            [format_record(hit, "match") for hit in result["hits"]]
            + [
                format_record(record, f"linked to {record['via']}")
                for record in result["graph_context"]
            ]
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("data_dir")
    build_parser.add_argument("db_dir")
    build_parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("db_dir")
    query_parser.add_argument("text")
    query_parser.add_argument("-k", type=int, default=5)
    query_parser.add_argument("--hops", type=int, default=1)
    query_parser.add_argument("--where")
    args = parser.parse_args(argv)

    if args.cmd == "build":
        build(args.data_dir, args.db_dir, args.embedder)
        return

    result = RCWGraphRAG(args.db_dir).retrieve(
        args.text, args.k, args.hops, where=args.where
    )
    for hit in result["hits"]:
        print(
            f"MATCH  {hit['id']:<16} {hit['section_title'][:70]}  "
            f"(dist {hit['_distance']:.3f})"
        )
    for record in result["graph_context"]:
        print(
            f"LINKED {record['id']:<16} {record['section_title'][:60]}  "
            f"<- {record['via']} ({record['hops']} hop)"
        )
