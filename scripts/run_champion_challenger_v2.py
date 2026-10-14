#!/usr/bin/env python3
from alpha_factor_engine.v2_pipeline import run_champion_challenger_v2


if __name__ == "__main__":
    outputs = run_champion_challenger_v2("configs/champion_challenger_v2.yml")
    print(f"report={outputs['report']}")
    print(f"manifest={outputs['manifest']}")
    print(f"governance_dir={outputs['governance_dir']}")
