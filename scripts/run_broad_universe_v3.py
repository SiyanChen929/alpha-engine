#!/usr/bin/env python3
from alpha_factor_engine.v3_pipeline import run_broad_universe_v3


if __name__ == "__main__":
    outputs = run_broad_universe_v3("configs/broad_universe_v3.yml")
    print(f"status={outputs['status']}")
    print(f"report={outputs['report']}")
    print(f"manifest={outputs['manifest']}")

