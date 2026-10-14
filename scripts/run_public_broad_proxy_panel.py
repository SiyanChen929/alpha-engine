#!/usr/bin/env python3
from proxy_research.build_panel import build_proxy_panel


if __name__ == "__main__":
    print(f"panel={build_proxy_panel('configs/public_broad_proxy_v1.yml')}")

