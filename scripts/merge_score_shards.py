#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge shard JSONL score files into a single file.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_handle:
        for input_name in args.input_jsonl:
            input_path = Path(input_name).resolve()
            if not input_path.exists():
                raise FileNotFoundError(f"Missing shard file: {input_path}")
            with input_path.open(encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)

    print(f"[rec:qwen] merged_shards={len(args.input_jsonl)} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
