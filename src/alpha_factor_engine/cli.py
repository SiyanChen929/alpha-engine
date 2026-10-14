"""Command-line interface."""

from __future__ import annotations

import argparse

from alpha_factor_engine.champion_pipeline import run_champion_challenger
from alpha_factor_engine.data.licensed import CONTRACTS, ParquetPITAdapter
from alpha_factor_engine.pipeline import run_public_research
from alpha_factor_engine.v2_pipeline import run_champion_challenger_v2


def main() -> None:
    parser = argparse.ArgumentParser(prog="alpha-factor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    public = subparsers.add_parser("public-research", help="run the Yahoo + SEC + FRED study")
    public.add_argument("--config", default="configs/public_research.yml")
    public.add_argument("--refresh", action="store_true")
    champion = subparsers.add_parser(
        "champion-challenger",
        help="run the opened momentum/ML comparison and freeze its prospective protocol",
    )
    champion.add_argument("--config", default="configs/champion_challenger.yml")
    champion_v2 = subparsers.add_parser(
        "champion-challenger-v2",
        help="run diversified momentum and the residual-ML overlay, then seal v2",
    )
    champion_v2.add_argument("--config", default="configs/champion_challenger_v2.yml")
    validate = subparsers.add_parser("validate-licensed", help="validate a licensed PIT parquet")
    validate.add_argument("--source", required=True, choices=sorted(CONTRACTS))
    validate.add_argument("--path", required=True)
    args = parser.parse_args()
    if args.command == "public-research":
        outputs = run_public_research(args.config, refresh=args.refresh)
        print(f"report={outputs['report']}")
        print(f"manifest={outputs['manifest']}")
    elif args.command == "champion-challenger":
        outputs = run_champion_challenger(args.config)
        print(f"report={outputs['report']}")
        print(f"manifest={outputs['manifest']}")
        print(f"governance_dir={outputs['governance_dir']}")
    elif args.command == "champion-challenger-v2":
        outputs = run_champion_challenger_v2(args.config)
        print(f"report={outputs['report']}")
        print(f"manifest={outputs['manifest']}")
        print(f"governance_dir={outputs['governance_dir']}")
    elif args.command == "validate-licensed":
        frame = ParquetPITAdapter(CONTRACTS[args.source]).load(args.path)
        print(f"source={args.source} rows={len(frame)} status=valid")


if __name__ == "__main__":
    main()
