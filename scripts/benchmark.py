"""合成性能基线命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.benchmark import run_benchmark


def main() -> None:
    """执行参数化基准并将结果打印或保存为 JSON。"""
    parser = argparse.ArgumentParser(description="Run the synthetic EvalKit performance baseline.")
    parser.add_argument("--cases", type=int, default=20, help="cases per measured run")
    parser.add_argument("--rounds", type=int, default=5, help="number of measured runs")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel independent runs")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured warmup runs")
    parser.add_argument("--output", type=Path, help="optional UTF-8 JSON output path")
    arguments = parser.parse_args()
    result = run_benchmark(
        case_count=arguments.cases,
        rounds=arguments.rounds,
        concurrency=arguments.concurrency,
        warmup_rounds=arguments.warmup,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
