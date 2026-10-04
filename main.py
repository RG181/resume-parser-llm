"""Command-line interface.

  python main.py parse data/samples/resume_01.txt --mode hybrid [--anonymize] [--out out.json]
  python main.py match data/samples/resume_01.txt --jd data/samples/jd_data_scientist.txt
  python main.py check      # verify your API key + model with one tiny request (run this first!)
  python main.py models     # list the models your API key can use
"""
import argparse
import json
import logging
import sys

from resume_parser import JobMatcher, LLMError, ResumeParser, to_json
from resume_parser.text_extraction import ExtractionError


def _utf8_console():
    """Windows consoles/pipes may use a legacy code page; make printing '–' or '•' safe."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None) -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(description="Hybrid NLP + LLM resume parser")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("parse", "match"):
        p = sub.add_parser(name)
        p.add_argument("resume", help="PDF / DOCX / TXT file")
        p.add_argument("--mode", choices=["rules", "llm", "hybrid"], help="override config pipeline.mode")
        p.add_argument("--anonymize", action="store_true", help="mask name/contact/college/years in the output")
        p.add_argument("--config", help="path to an alternative config.yaml")
        p.add_argument("-v", "--verbose", action="store_true")
    for name, helptext in (("check", "test API key + model"), ("models", "list available models")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("--config")
        q.add_argument("-v", "--verbose", action="store_true")
    sub.choices["parse"].add_argument("--out", help="write JSON here instead of stdout")
    sub.choices["match"].add_argument("--jd", required=True, help="job description text file")
    sub.choices["match"].add_argument("--no-explain", action="store_true", help="skip the LLM explanation")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    from resume_parser import LLMClient, load_config, load_prompts
    cfg = load_config(args.config)
    if args.cmd in {"check", "models"}:
        import time
        try:
            llm = LLMClient(cfg, load_prompts(cfg))
            print(f"provider={llm.provider}  model={llm.model}")
            if args.cmd == "models":
                print("\n".join(llm.list_models()))
            else:
                t0 = time.perf_counter()
                reply = llm.ping()
                print(f"OK - model replied {reply!r} in {time.perf_counter() - t0:.1f}s. You are ready to run `python main.py parse ...`")
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    parser = ResumeParser(cfg)
    try:
        resume = parser.parse_file(args.resume, mode=args.mode, anonymize_output=args.anonymize or None)
        if args.cmd == "parse":
            out = to_json(resume)
            if args.out:
                open(args.out, "w", encoding="utf-8").write(out)
                print(f"saved {args.out}")
            else:
                print(out)
        else:
            jd = open(args.jd, encoding="utf-8").read()
            matcher = JobMatcher(parser.cfg, parser.prompts, parser.taxonomy,
                                 parser.llm if not args.no_explain else None)
            print(json.dumps(matcher.match(resume, jd, explain=not args.no_explain), indent=2, ensure_ascii=False))
    except (ExtractionError, LLMError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
