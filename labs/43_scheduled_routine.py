"""Scheduled Cloud Routines — Session 43: Idempotent Task Runner.

Runs a Claude task and persists the result so the same task is never
re-run for the same day (idempotent). Designed to be invoked by cron
or 43_cron_agent.py.

Usage:
    python 43_scheduled_routine.py --task "Summarise today's top AI news" --name daily-ai-news
    python 43_scheduled_routine.py --task "..." --name my-task --force   # re-run even if done today
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

MODEL = "claude-opus-4-7"
OUTPUT_DIR = Path("./outputs/scheduled")

SYSTEM_PROMPT = """You are a research and analysis assistant executing a scheduled task.
Complete the task fully and return well-structured output (markdown or JSON as appropriate).
Be concise but complete. Include sources or reasoning where relevant."""


def result_path(task_name: str, run_date: date) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{run_date.isoformat()}-{task_name}.json"


def already_ran(task_name: str, run_date: date) -> bool:
    return result_path(task_name, run_date).exists()


def run_task(task_name: str, prompt: str) -> dict:
    llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096)

    started_at = datetime.utcnow().isoformat() + "Z"
    print(f"[{started_at}] {task_name} started")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    completed_at = datetime.utcnow().isoformat() + "Z"
    usage = response.usage_metadata or {}

    result = {
        "task_name": task_name,
        "prompt": prompt,
        "output": response.content,
        "started_at": started_at,
        "completed_at": completed_at,
        "tokens": {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        },
        "model": MODEL,
    }

    print(f"[{completed_at}] {task_name} completed ({usage.get('output_tokens', 0)} output tokens)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run an idempotent scheduled Claude task.")
    parser.add_argument("--task", required=True, help="The prompt / task to run")
    parser.add_argument("--name", required=True, help="Unique task name (used for dedup)")
    parser.add_argument("--force", action="store_true", help="Re-run even if already ran today")
    parser.add_argument("--output", default=None, help="Custom output path (overrides default)")
    args = parser.parse_args()

    today = date.today()
    out_path = Path(args.output) if args.output else result_path(args.name, today)

    if not args.force and out_path.exists():
        print(f"Already ran today: {out_path}. Use --force to re-run.")
        stored = json.loads(out_path.read_text())
        print(f"\nStored output:\n{stored['output']}")
        return

    result = run_task(args.name, args.task)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Result written to {out_path}")
    print(f"\n--- Output ---\n{result['output']}")


if __name__ == "__main__":
    main()
