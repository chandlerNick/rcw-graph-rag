#!/usr/bin/env python3
"""Crawl the unofficial RCW mirror at https://wa-law.org/rcw/ into GraphRAG-ready JSON.

Hierarchy:  root /rcw/  ->  title dirs (/rcw/01_general_provisions/)  ->  chapter pages (leaf, *.html, one <h2> per section).
Only wa-law.org HTML pages are fetched. apps.leg.wa.gov and every *.pdf (session-law PDFs) are never requested;
session-law citations are kept as parsed text only.

Outputs (in --out):
  titles.json, chapters.json   hierarchy metadata
  sections.jsonl               one record per RCW section (text + subsections + history + references)
  edges.jsonl                  graph edges (HAS_CHAPTER, HAS_SECTION, NEXT_SECTION, REFERENCES*, AMENDED_BY)
  crawl_report.json            counts, warnings, failures
"""
import argparse, hashlib, json, logging, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit, unquote
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("rcw")
BASE = "https://wa-law.org/rcw/"
ALIAS_HOSTS = {"wa-law.org", "www.wa-law.org"}
UA = "rcw-graphrag-crawler/1.0 (research use; polite)"
WS = re.compile(r"\s+")
clean = lambda t: WS.sub(" ", t.replace("\xa0", " ")).strip()

# ---------------------------------------------------------------- fetching
class Fetcher:
    def __init__(s, cache=None, delay=0.1, offline=None, timeout=30):
        s.cache = Path(cache) if cache else None; s.delay = delay; s.timeout = timeout
        s.offline = Path(offline) if offline else None; s.tl = threading.local()
        if s.cache: s.cache.mkdir(parents=True, exist_ok=True)
    def _sess(s):
        if not hasattr(s.tl, "s"):
            x = requests.Session(); x.headers["User-Agent"] = UA
            r = Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
            x.mount("http://", HTTPAdapter(max_retries=r)); x.mount("https://", HTTPAdapter(max_retries=r)); s.tl.s = x
        return s.tl.s
    def get(s, url):
        if url.lower().split("?")[0].endswith(".pdf"): raise ValueError(f"refusing PDF: {url}")
        if s.offline:  # read from a local checkout of tannewt/wa-law.org's site/ dir (zero network)
            p = s.offline / unquote(urlsplit(url).path).lstrip("/")
            return (p / "index.html" if p.is_dir() or url.endswith("/") else p).read_text("utf-8")
        cp = s.cache / (hashlib.sha1(url.encode()).hexdigest() + ".html") if s.cache else None
        if cp and cp.exists(): return cp.read_text("utf-8")
        time.sleep(s.delay)
        r = s._sess().get(url, timeout=s.timeout); r.raise_for_status()
        if "html" not in r.headers.get("content-type", ""): raise ValueError(f"non-HTML response from {url}: {r.headers.get('content-type')}")
        r.encoding = "utf-8"
        if cp: cp.write_text(r.text, "utf-8")
        return r.text

def norm_link(base, page_url, href):
    """Resolve href; return a canonical in-site URL (re-rooted on `base`) or None for external/PDF/anchor links."""
    if href.startswith("#"): return None
    u, b = urlsplit(urljoin(page_url, href)), urlsplit(base)
    if u.scheme not in ("http", "https") or u.path.lower().endswith(".pdf"): return None
    if u.netloc != b.netloc and u.netloc not in ALIAS_HOSTS: return None
    if not u.path.startswith(b.path.rstrip("/") + "/"): return None
    return urlunsplit((b.scheme, b.netloc, u.path, "", ""))

# ---------------------------------------------------------------- parsing helpers
def container(html):
    soup = BeautifulSoup(html, "lxml"); c = soup.select_one("div.container-sm") or soup.body
    for a in c.select("a.header-anchor"): a.decompose()
    return c

def split_footer(c):  # drop trailing <hr/> + "Created by @tannewt..." so it can't leak into the last section
    hrs = c.find_all("hr", recursive=False)
    if hrs and "Created by" in clean(" ".join(getattr(x, "get_text", lambda: str(x))() for x in hrs[-1].next_siblings)):
        for x in list(hrs[-1].next_siblings): x.extract()
        hrs[-1].extract()

def roman(n):
    out = ""
    for v, r in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v: out += r; n -= v
    return out

def fmt_label(kind, n):  # <ol type="a|A|i|I|None" start=n> -> (1) / (a) / (i) / (A); letters double after z like WA style (aa)
    if kind in ("a", "A"):
        s = chr(97 + (n - 1) % 26) * ((n - 1) // 26 + 1); return s.upper() if kind == "A" else s
    if kind in ("i", "I"): return roman(n).upper() if kind == "I" else roman(n)
    return str(n)

def pre_text(el): return "\n".join(l.rstrip() for l in el.get_text().strip("\n").splitlines())

def walk_list(ol, prefix, depth, out):
    kind, n = ol.get("type"), int(ol.get("start", 1) or 1)
    for li in ol.find_all("li", recursive=False):
        path = prefix + f"({fmt_label(kind, n)})"; n += 1; buf = []; first = [True]
        def flush():
            t = clean("".join(buf)); buf.clear()
            if t: out.append({"path": path, "depth": depth + 1, "text": t, **({} if first[0] else {"continuation": True})}); first[0] = False
        for ch in li.children:
            if isinstance(ch, Tag) and ch.name in ("ol", "ul"): flush(); walk_list(ch, path, depth + 1, out)
            elif isinstance(ch, Tag) and ch.name == "pre": flush(); out.append({"path": path, "depth": depth + 1, "text": pre_text(ch), "preformatted": True}); first[0] = False
            elif isinstance(ch, Tag) and ch.name == "p": flush(); buf.append(ch.get_text()); flush()
            else: buf.append(ch.get_text() if isinstance(ch, Tag) else str(ch))
        flush()

CITE = re.compile(r"^(\d{4}) ((?:\d+(?:st|nd|rd|th) )?(?:sp\.s\.|ex\.s\.) )?c (\d+) § (.+)$")
def parse_cite(item):
    m = CITE.match(item)
    if not m: return {"cite": item}
    y, sess, ch, sec = m.groups(); sess = sess.strip() if sess else None
    return {"cite": item, "year": int(y), "session": sess, "chapter": int(ch), "section": sec,
            "law_id": f"{y} {sess + ' ' if sess else ''}c {ch}"}

def is_history(p):
    t = clean(p.get_text())
    return t.startswith("[") and t.endswith("]") and t != "[Empty]" and bool(re.match(r"^\[\s*\d{4}\b", t) or p.find("a") or re.fullmatch(r"\[[\s;]*(NOTES:)?[\s;]*\]", t))

def parse_history(p):
    inner = clean(p.get_text()).strip("[] ")
    return [parse_cite(clean(x)) for x in inner.split(";") if clean(x) and clean(x) != "NOTES:"]

NOTE = re.compile(r"\s*\(((?:Effective|Expires|Contingent|Operative)[^()]*?)\)\.?$")  # e.g. "Intent. (Effective until January 1, 2022.)"

# cross-references: "RCW 9A.04.110(1)(a), 9.94A.505 and 9.94A.525", "chapter 70.02 RCW", "Title 9A RCW"
SEC = r"\d{1,3}[A-Z]{0,2}\.\d{1,3}[A-Z]{0,2}\.\d{1,4}[A-Z]{0,2}|\d{1,3}A\.\d{1,2}[A-Z]?-\d{1,4}[A-Z]?"
CHP = r"\d{1,3}[A-Z]{0,2}\.\d{1,3}[A-Z]{0,2}"; TTL = r"\d{1,3}[A-Z]{0,2}"
SEP = r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and/or|and\s+or|and|or|through|to)\s+)"
SUBS = r"(?:\s*\([A-Za-z0-9]{1,4}\))*"
RE_SEC = re.compile(rf"\bRCW\s+((?:{SEC}){SUBS}(?:{SEP}(?:{SEC}){SUBS})*)(?![\w-])")
RE_CHP = re.compile(rf"\b[Cc]hapters?\s+((?:{CHP})(?:{SEP}(?:{CHP}))*)\s+RCW\b")
RE_TTL = re.compile(rf"\b[Tt]itles?\s+((?:{TTL})(?:{SEP}(?:{TTL}))*)\s+RCW\b")
def uniq(xs): return list(dict.fromkeys(xs))
def extract_refs(text):
    secs = uniq(m for g in RE_SEC.findall(text) for m in re.findall(SEC, g))
    chps = uniq(m for g in RE_CHP.findall(text) for m in re.findall(CHP, g))
    ttls = uniq(m for g in RE_TTL.findall(text) for m in re.findall(rf"(?<![\w.]){TTL}(?![\w.])", g))
    return secs, chps, ttls

# ---------------------------------------------------------------- page parsers
def parse_index(html, url, base):
    """Root or title page -> (h1 text, [(abs_url, link_text)])"""
    c = container(html); h1 = clean(c.h1.get_text()) if c.h1 else ""
    page = url if url.endswith(("/", ".html")) else url + "/"
    return h1, [(u, clean(a.get_text())) for a in c.select("ul li a[href]") if (u := norm_link(base, page, a["href"]))]

def parse_chapter(html, url, title):
    c = container(html); split_footer(c); h1 = c.find("h1"); ht = clean(h1.get_text())
    m = re.match(r"^(\S+) - (.+)$", ht); cno, cname = (m.group(1), m.group(2)) if m else (ht, "")
    toc = h1.find_next_sibling()
    toc_n = len(toc.find_all("li", recursive=False)) if toc is not None and toc.name == "ul" else None
    chapter = {"chapter_number": cno, "chapter_name": cname, "title_number": title["title_number"], "title_name": title["title_name"], "url": url}
    secs, warns = [], []
    for h in c.find_all("h2"):
        hm = re.match(r"^(\S+) - (.+)$", clean(h.get_text()))
        if not hm: warns.append(f"{url}: unparseable h2 {clean(h.get_text())!r}"); continue
        num, name = hm.groups(); units, hist, intro = [], [], False
        for el in h.next_siblings:
            if not isinstance(el, Tag): continue
            if el.name == "h2": break
            if el.name in ("ol", "ul"): walk_list(el, "", 0, units)
            elif el.name == "pre": units.append({"path": "", "depth": 0, "text": pre_text(el), "preformatted": True})
            elif el.name == "p":
                if is_history(el): hist = parse_history(el)
                elif clean(el.get_text()): units.append({"path": "", "depth": 0, "text": clean(el.get_text())})
            elif el.name == "hr": units.append({"path": "", "depth": 0, "text": "---"})
            elif clean(el.get_text()): units.append({"path": "", "depth": 0, "text": clean(el.get_text())})
        seen_path = set(); lines = []
        for u in units:  # flat text: label only on the first unit of each subsection, indent by depth
            lab = "" if u["path"] in seen_path or not u["path"] else re.findall(r"\([^)]*\)", u["path"])[-1] + " "
            seen_path.add(u["path"]); lines.append("  " * u["depth"] + lab + u["text"])
        text = "\n".join(lines)
        sec, ch, tt = extract_refs(text)
        years = [x["year"] for x in hist if "year" in x]
        nm = NOTE.search(name)
        secs.append({
            "id": f"RCW {num}", "section_number": num, "section_title": name, "text": text, "subsections": units,
            "effective_note": nm.group(1).strip() if nm else None,
            "history": hist, "first_enacted_year": min(years) if years else None, "last_amended_year": max(years) if years else None,
            "references": [x for x in sec if x != num], "chapter_references": [x for x in ch if x != cno], "title_references": tt,
            "chapter_number": cno, "chapter_name": cname, "title_number": title["title_number"], "title_name": title["title_name"],
            "url": f"{url}#{h.get('id', '')}" if h.get("id") else url, "source_url": url,
            "word_count": len(text.split()), "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "embedding_text": f"RCW {num} — {name}\nTitle {title['title_number']} {title['title_name']} > Chapter {cno} {cname}\n\n{text}"})
    chapter["section_count"] = len(secs); chapter["toc_count"] = toc_n
    if toc_n is not None and toc_n != len(secs): warns.append(f"{url}: TOC lists {toc_n} sections but {len(secs)} parsed")
    if not secs: warns.append(f"{url}: no sections parsed")
    ids = [s["section_number"] for s in secs]
    if len(ids) != len(set(ids)): warns.append(f"{url}: duplicate section numbers")
    return chapter, secs, warns

# ---------------------------------------------------------------- orchestration
def crawl(a):
    base = a.base_url if a.base_url.endswith("/") else a.base_url + "/"
    f = Fetcher(None if a.no_cache else Path(a.out) / ".cache", a.delay, a.offline); t0 = time.time()
    _, tlinks = parse_index(f.get(base), base, base)
    titles, seen = [], set()
    for u, txt in tlinks:
        k = unquote(u)
        if u.endswith(".html") or k in seen or k.rstrip("/") == unquote(base).rstrip("/"): continue
        seen.add(k); titles.append((u if u.endswith("/") else u + "/", txt))
    want = {x.strip().upper() for x in a.titles.split(",")} if a.titles else None
    log.info("found %d titles", len(titles))
    chapters_todo, title_recs, failures = [], [], []
    def load_title(tu):
        u, txt = tu
        if want and txt.split(" - ")[0].strip().upper() not in want: return None, []  # skip fetching unwanted titles
        try:
            h1, links = parse_index(f.get(u), u, base); m = re.match(r"^(\S+)\s+(.*)$", h1)
            tr = {"title_number": m.group(1) if m else h1, "title_name": m.group(2) if m else "", "url": u, "chapter_numbers": []}
            return tr, [(cu, ct) for cu, ct in links if cu.endswith(".html")]
        except Exception as e: failures.append({"url": u, "error": repr(e)}); return None, []
    with ThreadPoolExecutor(a.workers) as ex: loaded = list(ex.map(load_title, titles))
    for tr, cl in loaded:
        if not tr or (want and tr["title_number"].upper() not in want): continue
        title_recs.append(tr); chapters_todo += [(tr, cu) for cu, _ in dict.fromkeys(cl)]
    if a.limit_chapters: chapters_todo = chapters_todo[:a.limit_chapters]
    log.info("fetching %d chapter pages", len(chapters_todo)); done = [0]
    def load_chapter(item):
        tr, cu = item
        try: r = parse_chapter(f.get(cu), cu, tr)
        except Exception as e: failures.append({"url": cu, "error": repr(e)}); r = None
        done[0] += 1
        if done[0] % 100 == 0: log.info("  %d/%d chapters (%.0fs)", done[0], len(chapters_todo), time.time() - t0)
        return tr, r
    with ThreadPoolExecutor(a.workers) as ex: results = list(ex.map(load_chapter, chapters_todo))
    chapters, sections, warns = [], [], []
    for tr, r in results:
        if not r: continue
        ch, secs, w = r; chapters.append(ch); sections += secs; warns += w; tr["chapter_numbers"].append(ch["chapter_number"])
    write_outputs(Path(a.out), title_recs, chapters, sections, warns, failures, t0, base)
    return 1 if failures else 0

def write_outputs(out, titles, chapters, sections, warns, failures, t0, base):
    out.mkdir(parents=True, exist_ok=True)
    jd = lambda o: json.dumps(o, ensure_ascii=False)
    sec_ids = {s["section_number"] for s in sections}; chap_ids = {c["chapter_number"] for c in chapters}; tit_ids = {t["title_number"] for t in titles}
    edges = []
    for t in titles: edges += [{"source": f"title:{t['title_number']}", "target": f"chapter:{c}", "type": "HAS_CHAPTER"} for c in t["chapter_numbers"]]
    prev = {}
    for s in sections:
        sid, cn = s["id"], s["chapter_number"]
        edges.append({"source": f"chapter:{cn}", "target": sid, "type": "HAS_SECTION"})
        if cn in prev: edges.append({"source": prev[cn], "target": sid, "type": "NEXT_SECTION"})
        prev[cn] = sid
        edges += [{"source": sid, "target": f"RCW {r}", "type": "REFERENCES", "resolved": r in sec_ids} for r in s["references"]]
        edges += [{"source": sid, "target": f"chapter:{r}", "type": "REFERENCES_CHAPTER", "resolved": r in chap_ids} for r in s["chapter_references"]]
        edges += [{"source": sid, "target": f"title:{r}", "type": "REFERENCES_TITLE", "resolved": r in tit_ids} for r in s["title_references"]]
        edges += [{"source": sid, "target": f"law:{h['law_id']}", "type": "AMENDED_BY", "cite": h["cite"]} for h in s["history"] if "law_id" in h]
    (out / "titles.json").write_text(json.dumps(titles, ensure_ascii=False, indent=1), "utf-8")
    (out / "chapters.json").write_text(json.dumps(chapters, ensure_ascii=False, indent=1), "utf-8")
    (out / "sections.jsonl").write_text("".join(jd(s) + "\n" for s in sections), "utf-8")
    (out / "edges.jsonl").write_text("".join(jd(e) + "\n" for e in edges), "utf-8")
    ref_e = [e for e in edges if e["type"] == "REFERENCES"]
    rep = {"crawled_at": datetime.now(timezone.utc).isoformat(), "base_url": base, "seconds": round(time.time() - t0, 1),
           "titles": len(titles), "chapters": len(chapters), "sections": len(sections), "edges": len(edges),
           "sections_with_effective_note": sum(bool(s["effective_note"]) for s in sections), "sections_without_history": sum(not s["history"] for s in sections),
           "section_refs": len(ref_e), "section_refs_unresolved": sum(not e["resolved"] for e in ref_e),
           "duplicate_section_ids": len(sections) - len(sec_ids), "failures": failures, "warnings": warns}
    (out / "crawl_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), "utf-8")
    log.info("done: %d titles, %d chapters, %d sections, %d edges, %d failures, %d warnings -> %s", len(titles), len(chapters), len(sections), len(edges), len(failures), len(warns), out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="rcw_data"); p.add_argument("--base-url", default=BASE)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--delay", type=float, default=0.1, help="seconds before each uncached request, per worker")
    p.add_argument("--titles", help="comma list, e.g. 1,9A,70A"); p.add_argument("--limit-chapters", type=int)
    p.add_argument("--no-cache", action="store_true", help="don't cache raw HTML under <out>/.cache (cache makes reruns/resume free)")
    p.add_argument("--offline", help="read pages from a local checkout of tannewt/wa-law.org's site/ dir instead of the network")
    a = p.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sys.exit(crawl(a))