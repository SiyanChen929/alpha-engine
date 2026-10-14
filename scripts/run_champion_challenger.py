#!/usr/bin/env python3
from alpha_factor_engine.champion_pipeline import run_champion_challenger


if __name__ == "__main__":
    outputs = run_champion_challenger("configs/champion_challenger.yml")
    print(f"report={outputs['report']}")
    print(f"manifest={outputs['manifest']}")
    print(f"governance_dir={outputs['governance_dir']}")
