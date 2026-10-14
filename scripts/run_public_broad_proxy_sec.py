#!/usr/bin/env python3
from proxy_research.acquire_sec import acquire_sec


if __name__ == "__main__":
    result = acquire_sec("configs/public_broad_proxy_v1.yml")
    for key, value in result.items():
        print(f"{key}={value}")

