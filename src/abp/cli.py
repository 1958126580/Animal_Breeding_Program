"""Command-line interface: ``abp <command> ...``.

Exit statuses are stable (see :mod:`abp.errors`): 0 success, 2 usage,
3 input, 4 QC blocked, 5 model, 6 numerical, 7 resource, 130 cancelled,
1 internal error.  Messages go to stderr; results go to the output folder.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from . import __version__
from .errors import ERROR_SPECS, EXIT_OK, EXIT_USAGE, ABPError


def _utf8_console() -> None:
    """Make stdout/stderr UTF-8 so Chinese paths and IDs print on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _sigterm_to_interrupt(signum, frame):  # pragma: no cover - signal path
    raise KeyboardInterrupt


def cmd_run(args) -> int:
    from .workflows.evaluate import run_evaluation
    out = run_evaluation(args.spec, args.out, force=args.force, resume=args.resume,
                         console=not args.quiet)
    print(f"status: {out.status}")
    print(f"outputs: {out.out_dir}")
    print(f"report: {out.out_dir / 'report.md'}")
    return EXIT_OK


def cmd_validate(args) -> int:
    """Validate the spec and run all QC without fitting any model."""
    from .workflows.validate import validate_inputs
    summary = validate_inputs(args.spec)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return EXIT_OK


def cmd_pedigree(args) -> int:
    from .workflows.pedigree_tools import pedigree_report
    out = pedigree_report(args.pedigree, args.out, id_col=args.id, sire_col=args.sire,
                          dam_col=args.dam, sex_col=args.sex, birth_col=args.birth_date,
                          delimiter=args.delimiter, force=args.force)
    print(f"outputs: {out}")
    return EXIT_OK


def cmd_index(args) -> int:
    from .decision.index_cli import run_index_spec
    res = run_index_spec(args.spec)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return EXIT_OK


def cmd_simulate(args) -> int:
    from .examples.sheep import write_sheep_example
    out = write_sheep_example(Path(args.out), seed=args.seed, force=args.force)
    print(f"synthetic sheep example written to: {out}")
    return EXIT_OK


def cmd_selftest(args) -> int:
    from .selftest import run_selftest
    ok, lines = run_selftest()
    for line in lines:
        print(line)
    return EXIT_OK if ok else 6


def cmd_errors(args) -> int:
    for spec in ERROR_SPECS.values():
        print(f"{spec.code}  exit={spec.exit_status:<3} {spec.name}: {spec.summary}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abp", description="ABP: auditable genetic evaluation for "
                                "animal breeding (pedigree/genomic BLUP, REML, selection index).")
    p.add_argument("--version", action="version", version=f"abp {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a complete evaluation from an analysis spec (TOML)")
    r.add_argument("spec", help="analysis spec file (.toml)")
    r.add_argument("--out", required=True, help="output folder (created; must be empty)")
    r.add_argument("--force", action="store_true", help="replace an existing output folder atomically")
    r.add_argument("--resume", action="store_true", help="resume REML from a matching checkpoint")
    r.add_argument("--quiet", action="store_true", help="log to run.log only")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="validate spec and data (QC only, no model fitting)")
    v.add_argument("spec")
    v.set_defaults(func=cmd_validate)

    pd = sub.add_parser("pedigree", help="pedigree QC, inbreeding and A-inverse statistics")
    pd.add_argument("pedigree", help="pedigree CSV")
    pd.add_argument("--out", required=True)
    pd.add_argument("--id", default="id")
    pd.add_argument("--sire", default="sire")
    pd.add_argument("--dam", default="dam")
    pd.add_argument("--sex", default=None)
    pd.add_argument("--birth-date", default=None)
    pd.add_argument("--delimiter", default=",")
    pd.add_argument("--force", action="store_true")
    pd.set_defaults(func=cmd_pedigree)

    ix = sub.add_parser("index", help="compute a Smith-Hazel selection index from a TOML spec")
    ix.add_argument("spec")
    ix.set_defaults(func=cmd_index)

    sm = sub.add_parser("simulate-sheep", help="write the synthetic sheep example data set")
    sm.add_argument("--out", required=True)
    sm.add_argument("--seed", type=int, default=20260925)
    sm.add_argument("--force", action="store_true")
    sm.set_defaults(func=cmd_simulate)

    st = sub.add_parser("selftest", help="run built-in analytical checks (installation test)")
    st.set_defaults(func=cmd_selftest)

    er = sub.add_parser("errors", help="list error codes and exit statuses")
    er.set_defaults(func=cmd_errors)
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    try:
        signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    except (ValueError, AttributeError):  # pragma: no cover - non-main thread / platform
        pass
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    try:
        return args.func(args)
    except ABPError as err:
        print(f"error {err.code} ({err.spec.name}): {err.message}", file=sys.stderr)
        print(f"remedy: {err.spec.remedy}", file=sys.stderr)
        findings = err.details.get("findings")
        if findings:
            for f in findings[:10]:
                ex = "; ".join(json.dumps(e, ensure_ascii=False) for e in f["examples"][:3])
                print(f"  - {f['check']}: {f['message']} [{f['count']}] {ex}", file=sys.stderr)
        return err.exit_status
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
