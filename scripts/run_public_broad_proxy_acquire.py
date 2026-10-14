#!/usr/bin/env python3
from proxy_research.acquire_prices import acquire_prices


if __name__ == "__main__":
    result = acquire_prices("configs/public_broad_proxy_v1.yml")
    for key, value in result.items():
        print(f"{key}={value}")

