#!/usr/bin/env python
from proxy_research.robustness import run_random_universes


if __name__ == "__main__":
    print(run_random_universes("configs/public_broad_proxy_v1.yml"))
