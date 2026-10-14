#!/usr/bin/env python
from proxy_research.robustness import run_label_permutations


if __name__ == "__main__":
    print(run_label_permutations("configs/public_broad_proxy_v1.yml"))
