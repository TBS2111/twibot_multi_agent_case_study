"""Stage 2: retrieve guidance / ATT&CK notes. Keyword (SQLite FTS5, bm25) search, no embeddings needed.

Retrieved notes are BACKGROUND ONLY. They are rendered in their own report section and the
validator refuses to let them be cited as evidence about the dataset.
"""
import json
import re
import sqlite3
from pathlib import Path

NOTES = Path(__file__).resolve().parent / "knowledge" / "notes.json"

# evidence kind -> search words (deterministic query construction)
KIND_TERMS = {
    "creation_burst": "account creation registered batch",
    "creation_concentration": "account creation registered batch",
    "shared_url": "shared links url promotion",
    "shared_hashtag": "hashtag amplification coordinated",
    "template_tweet": "template identical repeated content",
    "cohesion": "coordinated similarity network",
    "profile": "profile fake accounts social media",
}


class KnowledgeBase:
    def __init__(self, path=NOTES):
        self.notes = {n["id"]: n for n in json.loads(Path(path).read_text(encoding="utf-8"))}
        self.con = sqlite3.connect(":memory:")
        self.con.execute("CREATE VIRTUAL TABLE fts USING fts5(id UNINDEXED, title, text, tokenize='porter')")
        self.con.executemany("INSERT INTO fts VALUES (?,?,?)",
                             [(i, n["title"], n["text"]) for i, n in sorted(self.notes.items())])

    def search(self, terms: str, k: int = 4):
        words = sorted(set(re.findall(r"[a-z]+", terms.lower())))
        if not words:
            return []
        q = " OR ".join(words)
        rows = self.con.execute(
            "SELECT id FROM fts WHERE fts MATCH ? ORDER BY bm25(fts), id LIMIT ?", (q, k)).fetchall()
        return [self.notes[r[0]] for r in rows]

    def for_evidence(self, evidence, k: int = 4):
        kinds = sorted({e.kind for e in evidence})
        terms = " ".join(KIND_TERMS.get(x, "") for x in kinds) + " social media accounts"
        return self.search(terms, k)
