"""python -m unittest investigation.tests.test_agents -v

A ScriptedLLM stands in for Claude, so these tests check the AGENT BEHAVIOUR (planning, revise loop,
fallback, rejection of bad model output) with no network. They say nothing about real model quality.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from investigation import cli, ingest, report, timeline, validate
from investigation.llm import ScriptedLLM
from investigation.query_parser import Query
from investigation.tests.test_investigation import make_fixture
from investigation.workflow import run_investigation, run_sequential


def good_body(user):
    items = json.loads(re.search(r"EVIDENCE_JSON\n(.*?)\nEND_EVIDENCE_JSON", user, re.S).group(1))
    ev = timeline.evidence_from_json([dict(d, source=d["kind"]) for d in items])
    return "\n".join(["## Findings"] + report.findings(ev) + ["", "## Timeline", ""] + report.timeline_lines(ev, Query()))


def bad_body(user):
    return re.sub(r"has (\d+) accounts", lambda m: f"has {int(m.group(1)) + 1} accounts", good_body(user), count=1)


def make_llm(report_mode="bad_then_good", plan='{"communities":[0,1],"params":{"top_k":2},"reason":"bot groups"}'):
    seen = {"review": 0}

    def handler(system, user):
        if "ROLE:QUERY" in system:
            return '{"selector":"largest","top_n":2,"min_bot_ratio":0.8}'
        if "ROLE:PLANNER" in system:
            return plan
        if "ROLE:REVIEW" in system:
            seen["review"] += 1
            return '{"action":"deepen","params":{"top_k":3}}' if seen["review"] == 1 else '{"action":"done"}'
        if "ROLE:KEYWORDS" in system:
            return '{"keywords":["shared","links","account","creation"]}'
        if "ROLE:REPORT" in system:
            if report_mode == "always_bad" or (report_mode == "bad_then_good" and "VALIDATION_ISSUES" not in user):
                return bad_body(user)
            return good_body(user)
        raise AssertionError("unexpected role")
    return ScriptedLLM(handler)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        data, out = make_fixture(t)
        cls.db, cls.reports = t / "e.db", t / "r"
        ingest.build_db(out, data, cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_it(self, llm, **kw):
        return run_investigation("investigate the biggest bot groups", self.db, self.reports, llm, **kw)

    def test_llm_plans_and_revise_loop_fixes_hallucination(self):
        llm = make_llm("bad_then_good")
        st, path = self.run_it(llm)
        self.assertTrue(st["ok"])
        self.assertFalse(st["used_fallback"])
        self.assertEqual(st["revisions"], 2)                         # draft 1 rejected, draft 2 accepted
        self.assertEqual(st["communities"], [0, 1])                  # planner's choice
        self.assertEqual(st["params"].top_k, 3)                      # deepened by the review step
        self.assertTrue(any("[validator] FAIL" in l for l in st["log"]))
        self.assertTrue(any("VALIDATION_ISSUES" in u for s, u in llm.calls if "ROLE:REPORT" in s))
        self.assertEqual(cli.cmd_validate(path, self.db), [])        # sidecar reproduces from the DB

    def test_template_fallback_when_model_keeps_hallucinating(self):
        st, _ = self.run_it(make_llm("always_bad"), max_revisions=2)
        self.assertTrue(st["ok"])
        self.assertTrue(st["used_fallback"])
        self.assertEqual(st["revisions"], 4)                         # 3 LLM drafts + template

    def test_unknown_community_from_planner_is_rejected(self):
        st, _ = self.run_it(make_llm("good", plan='{"communities":[999],"params":{}}'))
        self.assertTrue(st["ok"])
        self.assertTrue(any("planner rejected" in l for l in st["log"]))
        self.assertNotIn(999, st["communities"])

    def test_garbage_model_output_degrades_to_deterministic(self):
        st, _ = self.run_it(ScriptedLLM(lambda s, u: "I cannot help with that"))
        self.assertTrue(st["ok"])
        self.assertTrue(st["used_fallback"])

    def test_no_llm_matches_deterministic_pipeline(self):
        st, _ = run_investigation("investigate community 0", self.db, self.reports, None)
        ok, _, md = cli.investigate("investigate community 0", self.db, self.reports)
        self.assertTrue(st["ok"] and ok)
        self.assertEqual(st["draft"], md)

    def test_sequential_runner_matches_graph_routing(self):
        st = run_sequential({"question": "investigate the biggest bot groups", "db": str(self.db),
                             "llm": make_llm("bad_then_good"), "max_revisions": 3, "log": []})
        self.assertTrue(st["ok"] and st["revisions"] == 2)

    def test_validator_blocks_injected_links_and_markup(self):
        con = ingest.connect(self.db)
        ev = timeline.collect(con, Query(community=0))
        notes = []
        md = report.render_template(Query(community=0), ev, notes, timeline.digest(ev))
        for payload in (" Visit https://evil.example now.", " [click](http://x.y)", " <script>x</script>"):
            bad = md.replace("[E1]", payload + " [E1]", 1)
            self.assertTrue(any(i.code == "format" for i in validate.validate(bad, ev, notes)), payload)


if __name__ == "__main__":
    unittest.main()
