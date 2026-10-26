"""Investigation pipeline CLI.

  python -m investigation.cli ingest   --out-dir outputs --data-dir data --db outputs/evidence.db
  python -m investigation.cli ask "investigate the 2 largest communities with bot ratio above 0.8" \
         --db outputs/evidence.db --report-dir outputs/reports [--llm-parse] [--llm-report]
  python -m investigation.cli agent "question" --db outputs/evidence.db     # LLM-driven agents
  python -m investigation.cli validate outputs/reports/report.md --db outputs/evidence.db

Flow of `ask`:  parse question -> SQL evidence + timeline -> FTS background notes
                -> template report (optional LLM rewrite) -> deterministic validation gate.
A report that fails the gate is written as *.REJECTED.md and the exit code is 2.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from investigation import ingest, query_parser, report, timeline, validate
from investigation.knowledge import KnowledgeBase


def investigate(question, db, report_dir, llm_parse=False, llm_report=False, params=None):
    params = params or timeline.Params()
    q = query_parser.parse(question, use_llm=llm_parse)
    con = ingest.connect(db)
    evidence = timeline.collect(con, q, params)
    con.close()
    if not evidence:
        raise SystemExit("No community matched the question (check min_size / bot ratio filters).")
    dig = timeline.digest(evidence)
    notes = KnowledgeBase().for_evidence(evidence)
    md = report.render_template(q, evidence, notes, dig)
    used_llm = False
    if llm_report:
        try:
            cand = report.llm_rewrite(md)
            if not validate.validate(cand, evidence, notes):
                md, used_llm = cand, True
            else:
                print("LLM rewrite failed validation; using the template report.", file=sys.stderr)
        except Exception as ex:                      # missing package/key/network: stay deterministic
            print(f"LLM rewrite unavailable ({ex}); using the template report.", file=sys.stderr)
    issues = validate.validate(md, evidence, notes)

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = not issues
    (out / ("report.md" if ok else "report.REJECTED.md")).write_text(md, encoding="utf-8")
    (out / "report.evidence.json").write_text(json.dumps(dict(
        query=asdict(q), params=asdict(params), digest=dig, communities=list(dict.fromkeys(e.community for e in evidence)), llm_rewrite=used_llm,
        evidence=[asdict(e) for e in evidence], notes=notes), indent=2, sort_keys=True), encoding="utf-8")
    return ok, issues, md


def cmd_validate(path, db):
    path = Path(path)
    side = json.loads(path.with_name("report.evidence.json").read_text(encoding="utf-8"))
    evidence = timeline.evidence_from_json(side["evidence"])
    issues = validate.validate(path.read_text(encoding="utf-8"), evidence, side["notes"], side["digest"])
    if db:                                           # reproducibility: re-derive evidence from the DB
        con = ingest.connect(db)
        q = query_parser.Query.from_dict(side["query"], raw=side["query"].get("raw", ""))
        again = timeline.collect(con, q, timeline.Params(**side["params"]), side.get("communities"))
        con.close()
        if timeline.digest(again) != side["digest"]:
            issues.append(validate.Issue(0, "reproducibility", "re-running the SQL gives different evidence"))
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser(prog="investigation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("--out-dir", default="outputs")
    a.add_argument("--data-dir", default="data")
    a.add_argument("--db", default="outputs/evidence.db")
    a.add_argument("--community-col", default="refined", choices=["refined", "initial"])
    b = sub.add_parser("ask")
    b.add_argument("question")
    b.add_argument("--db", default="outputs/evidence.db")
    b.add_argument("--report-dir", default="outputs/reports")
    b.add_argument("--llm-parse", action="store_true")
    b.add_argument("--llm-report", action="store_true")
    d = sub.add_parser("agent", help="LLM-driven autonomous investigation (needs langchain-anthropic + API key)")
    d.add_argument("question")
    d.add_argument("--db", default="outputs/evidence.db")
    d.add_argument("--report-dir", default="outputs/reports")
    d.add_argument("--model", default="claude-sonnet-4-6")
    d.add_argument("--max-revisions", type=int, default=3)
    d.add_argument("--no-llm", action="store_true", help="run the same agents fully deterministically")
    c = sub.add_parser("validate")
    c.add_argument("report")
    c.add_argument("--db", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        print(ingest.build_db(args.out_dir, args.data_dir, args.db, args.community_col))
        return 0
    if args.cmd == "agent":
        from investigation.workflow import run_investigation
        llm = None
        if not args.no_llm:
            from investigation.llm import ClaudeLLM
            llm = ClaudeLLM(args.model)
        st, path = run_investigation(args.question, args.db, args.report_dir, llm, args.max_revisions)
        print("\n".join(st["log"]), file=sys.stderr)
        print(st["draft"])
        print(f"{'VALIDATION PASSED' if st['ok'] else 'VALIDATION FAILED'} -> {path}", file=sys.stderr)
        return 0 if st["ok"] else 2
    if args.cmd == "ask":
        ok, issues, md = investigate(args.question, args.db, args.report_dir, args.llm_parse, args.llm_report)
        print(md)
        if not ok:
            print("VALIDATION FAILED:\n" + "\n".join(map(str, issues)), file=sys.stderr)
            return 2
        print("VALIDATION PASSED", file=sys.stderr)
        return 0
    issues = cmd_validate(args.report, args.db)
    print("VALIDATION PASSED" if not issues else "VALIDATION FAILED:\n" + "\n".join(map(str, issues)))
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
