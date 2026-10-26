"""Run from the project root:  python -m unittest investigation.tests.test_investigation -v

Builds a synthetic dataset, FAKES the pipeline outputs (communities.csv / graph.npz) so the test
needs no torch/langgraph, then exercises ingest -> evidence -> report -> validator.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from investigation import cli, ingest, query_parser, timeline, validate
from investigation.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parents[2]


def make_fixture(tmp: Path):
    data = tmp / "data"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), str(data)], check=True,
                   capture_output=True)
    rows, ids = [], []
    for split in ("train", "dev", "test"):
        for u in json.loads((data / f"{split}.json").read_text()):
            rows.append((str(u["ID"]), int(u["label"]), split))
    # community = botnet group g for ids u10000+g*100.., all humans in community 99
    comm = [((int(i[1:]) - 10000) // 100) if int(i[1:]) < 50000 else 99 for i, _, _ in rows]
    df = pd.DataFrame(rows, columns=["id", "label", "split"])
    df["initial"], df["refined"] = comm, comm
    out = tmp / "outputs"
    out.mkdir()
    df.to_csv(out / "communities.csv", index=False)
    rng = np.random.RandomState(0)
    edges, w, sims = [], [], []
    by_c = df.groupby("refined").indices
    for c, idx in by_c.items():
        for _ in range(len(idx) * 3):
            a, b = sorted(rng.choice(idx, 2, replace=False))
            edges.append((a, b)); w.append(0.7 if c != 99 else 0.1); sims.append(rng.rand(4))
    for _ in range(40):                              # a few negative boundary edges
        a, b = sorted(rng.choice(len(df), 2, replace=False))
        edges.append((a, b)); w.append(-0.3); sims.append(rng.rand(4))
    np.savez(out / "graph.npz", edges=np.array(edges), w=np.array(w), sims=np.array(sims))
    return data, out


class InvestigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        data, out = make_fixture(t)
        cls.db = t / "evidence.db"
        ingest.build_db(out, data, cls.db)
        cls.reports = t / "reports"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parser(self):
        q = query_parser.parse_rules("show the top 3 largest communities with bot ratio above 80% in 2020")
        self.assertEqual((q.top_n, q.selector, q.min_bot_ratio, q.since, q.until),
                         (3, "largest", 0.8, "2020-01-01", "2020-12-31"))
        self.assertEqual(query_parser.parse_rules("what is going on in community 4").community, 4)
        q = query_parser.parse_rules("investigate the largest bot community")
        self.assertEqual((q.selector, q.min_bot_ratio), ("largest", 0.5))
        with self.assertRaises(ValueError):
            query_parser.Query.from_dict({"since": "yesterday"})

    def test_pass_and_reproducible(self):
        ok1, issues1, md1 = cli.investigate("investigate the 2 largest communities with bot ratio above 0.8",
                                            self.db, self.reports)
        self.assertTrue(ok1, [str(i) for i in issues1])
        ok2, _, md2 = cli.investigate("investigate the 2 largest communities with bot ratio above 0.8",
                                      self.db, self.reports)
        self.assertEqual(md1, md2)                    # byte-identical
        self.assertEqual(cli.cmd_validate(self.reports / "report.md", self.db), [])

    def _ctx(self):
        con = ingest.connect(self.db)
        q = query_parser.parse_rules("investigate community 0")
        ev = timeline.collect(con, q)
        notes = KnowledgeBase().for_evidence(ev)
        from investigation import report
        md = report.render_template(q, ev, notes, timeline.digest(ev))
        return ev, notes, md

    def test_largest_bot_community_is_not_the_human_one(self):
        ok, _, md = cli.investigate("investigate the largest bot community", self.db, self.reports)
        self.assertTrue(ok)
        self.assertNotIn("### Community 99", md)

    def test_gate_catches_tampering(self):
        ev, notes, md = self._ctx()
        self.assertEqual(validate.validate(md, ev, notes), [])
        cases = {
            "number": md.replace("has 25 accounts", "has 26 accounts"),
            "technique": md + "\n- Mapped to T1566 phishing. [K:T1566]\n",
            "citation": md.replace("[E1]", "", 1),
            "string": md.replace("`u10000`", "`u99999`"),
            "overclaim": md.replace("signals to verify", "proves coordination"),
            "digest": md.replace(timeline.digest(ev), "0" * 16),
        }
        for code, bad in cases.items():
            with self.subTest(code):
                self.assertTrue(validate.validate(bad, ev, notes), f"{code} tampering was not caught")

    def test_tampered_evidence_file_fails_reproducibility(self):
        cli.investigate("investigate community 0", self.db, self.reports)
        side = self.reports / "report.evidence.json"
        d = json.loads(side.read_text())
        d["evidence"][0]["facts"]["size"] += 1
        side.write_text(json.dumps(d))
        self.assertTrue(cli.cmd_validate(self.reports / "report.md", self.db))

    def test_fts_is_deterministic(self):
        kb = KnowledgeBase()
        a = [n["id"] for n in kb.search("shared links url promotion")]
        self.assertEqual(a, [n["id"] for n in kb.search("shared links url promotion")])
        self.assertIn("G-SHARED-LINKS", a)


if __name__ == "__main__":
    unittest.main()
