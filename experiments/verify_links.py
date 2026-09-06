#!/usr/bin/env python
"""Verify external links in repository documentation."""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import requests


URL_RE = re.compile(r"https?://[^\s)>\"]+")


def extract_urls(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return sorted(set(URL_RE.findall(text)))


def main():
    parser = argparse.ArgumentParser(description="Verify links in markdown docs")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["README.md", "CONTRIBUTING.md", "PROJECT.md"],
    )
    parser.add_argument("--output", type=str, default="results/link_check.csv")
    args = parser.parse_args()

    rows = []
    for rel in args.files:
        if not os.path.exists(rel):
            continue
        for url in extract_urls(rel):
            status = None
            ok = False
            error = ""
            try:
                resp = requests.get(url, timeout=20, allow_redirects=True)
                status = resp.status_code
                ok = 200 <= status < 400
            except Exception as e:
                error = str(e)
            rows.append(
                {
                    "file": rel,
                    "url": url,
                    "status_code": status,
                    "ok": ok,
                    "error": error,
                }
            )

    import pandas as pd

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Link check written to {args.output}")


if __name__ == "__main__":
    main()
