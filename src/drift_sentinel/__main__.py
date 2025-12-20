import argparse
import json
from pathlib import Path
from .experiments import demo, evaluate


def main():
    parser = argparse.ArgumentParser(description="Sequential bounded-mean drift and delayed-label monitoring")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate", "metrics"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path)
        if name == "evaluate":
            command.add_argument("--trials", type=int, default=24)
        if name == "demo":
            command.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    result = evaluate(trials=args.trials) if args.command == "evaluate" else demo()
    text = (
        result["prometheus"]
        if args.command == "metrics"
        else json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.command == "demo" and args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(result["prometheus"])


if __name__ == "__main__":
    main()
