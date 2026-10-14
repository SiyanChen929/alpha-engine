#!/usr/bin/env python3
from alpha_factor_engine.pipeline import run_public_research


if __name__ == "__main__":
    outputs = run_public_research("configs/public_research.yml")
    print(f"report={outputs['report']}")

