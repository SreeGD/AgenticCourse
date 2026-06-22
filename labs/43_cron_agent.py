"""Scheduled Cloud Routines — Session 43: Cron Agent Registry.

Manages a registry of named routines (stored in routines.json) and
can register, list, run, and delete them. Routines are invoked by
43_scheduled_routine.py under the hood.

Usage:
    python 43_cron_agent.py register --name weekly-summary --prompt "..." --schedule "0 7 * * 1"
    python 43_cron_agent.py list
    python 43_cron_agent.py run --name weekly-summary
    python 43_cron_agent.py delete --name weekly-summary

To wire into system cron, add a line like:
    0 7 * * 1  cd /path/to/repo && python 43_cron_agent.py run --name weekly-summary
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REGISTRY_PATH = Path("./outputs/scheduled/routines.json")


def load_registry() -> dict:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def save_registry(registry: dict):
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def cmd_register(args):
    registry = load_registry()
    if args.name in registry and not args.force:
        print(f"Routine '{args.name}' already exists. Use --force to overwrite.")
        return

    registry[args.name] = {
        "name": args.name,
        "prompt": args.prompt,
        "schedule": args.schedule,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_run": None,
        "run_count": 0,
    }
    save_registry(registry)
    print(f"Registered routine '{args.name}' (schedule: {args.schedule})")
    print(f"\nTo wire into system cron, run: crontab -e")
    print(f"Add: {args.schedule}  cd {Path.cwd()} && python 43_cron_agent.py run --name {args.name}")


def cmd_list(args):
    registry = load_registry()
    if not registry:
        print("No routines registered.")
        return

    print(f"{'NAME':<25} {'SCHEDULE':<20} {'LAST RUN':<30} {'RUNS'}")
    print("-" * 85)
    for name, r in sorted(registry.items()):
        last = r.get("last_run") or "never"
        print(f"{name:<25} {r['schedule']:<20} {last:<30} {r['run_count']}")


def cmd_run(args):
    registry = load_registry()
    if args.name not in registry:
        print(f"Routine '{args.name}' not found. Run 'list' to see registered routines.")
        sys.exit(1)

    routine = registry[args.name]
    print(f"Running routine '{args.name}'...")

    result = subprocess.run(
        [
            sys.executable, "43_scheduled_routine.py",
            "--task", routine["prompt"],
            "--name", args.name,
        ] + (["--force"] if args.force else []),
        capture_output=False,
    )

    if result.returncode == 0:
        registry[args.name]["last_run"] = datetime.utcnow().isoformat() + "Z"
        registry[args.name]["run_count"] += 1
        save_registry(registry)
    else:
        print(f"Routine '{args.name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def cmd_delete(args):
    registry = load_registry()
    if args.name not in registry:
        print(f"Routine '{args.name}' not found.")
        return
    del registry[args.name]
    save_registry(registry)
    print(f"Deleted routine '{args.name}'.")


def main():
    parser = argparse.ArgumentParser(description="Manage scheduled Claude routines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register
    reg = subparsers.add_parser("register", help="Register a new routine")
    reg.add_argument("--name", required=True)
    reg.add_argument("--prompt", required=True)
    reg.add_argument("--schedule", required=True, help="Cron expression e.g. '0 7 * * 1'")
    reg.add_argument("--force", action="store_true", help="Overwrite existing routine")

    # list
    subparsers.add_parser("list", help="List all registered routines")

    # run
    run_p = subparsers.add_parser("run", help="Run a routine now")
    run_p.add_argument("--name", required=True)
    run_p.add_argument("--force", action="store_true", help="Re-run even if already ran today")

    # delete
    del_p = subparsers.add_parser("delete", help="Delete a routine")
    del_p.add_argument("--name", required=True)

    args = parser.parse_args()

    match args.command:
        case "register":
            cmd_register(args)
        case "list":
            cmd_list(args)
        case "run":
            cmd_run(args)
        case "delete":
            cmd_delete(args)


if __name__ == "__main__":
    main()
