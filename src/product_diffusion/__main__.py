"""Run the case study and write the figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_diffusion.plots import render
from product_diffusion.study import run_case_study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results/case_study.json"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--replot",
        type=Path,
        help="Render figures from a saved results file instead of rerunning the search.",
    )
    args = parser.parse_args(argv)

    if args.replot is not None:
        payload = json.loads(args.replot.read_text())
    else:
        payload = run_case_study(
            seed=args.seed,
            restarts=args.restarts,
            maxiter=args.maxiter,
            depths=tuple(args.depths),
        )
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.results}", flush=True)

    written = render(payload, args.figures)
    for path in written:
        print(path, flush=True)


if __name__ == "__main__":
    main()
