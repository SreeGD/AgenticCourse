"""Codebase Archaeology — Session 41: Repo Map Builder.

Walks a local repository, collects file sizes, detects entry points,
extracts top-level imports, and emits a structured JSON map that
41_architecture_summary.py sends to Claude for an architecture narrative.

Usage:
    python 41_repo_map.py --path /path/to/repo --output repo_map.json
    python 41_repo_map.py --path .              # defaults to stdout
"""

import argparse
import ast
import json
import os
from pathlib import Path

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "migrations", ".next", "coverage",
}
IGNORE_EXTS = {".pyc", ".pyo", ".lock", ".log", ".DS_Store"}

ENTRY_NAMES = {
    "main.py", "app.py", "server.py", "index.py",
    "manage.py", "cli.py", "run.py", "wsgi.py", "asgi.py",
    "index.ts", "index.js", "main.ts", "server.ts",
}

LANGUAGE_MAP = {
    ".py": "Python", ".ts": "TypeScript", ".js": "JavaScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
    ".cs": "C#", ".cpp": "C++", ".c": "C", ".md": "Markdown",
    ".sql": "SQL", ".sh": "Shell", ".yaml": "YAML", ".yml": "YAML",
    ".json": "JSON", ".toml": "TOML",
}


def extract_python_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        return sorted(set(imports))
    except Exception:
        return []


def walk_repo(root: Path) -> dict:
    root = root.resolve()
    files = []
    entry_points = []
    dir_summary: dict[str, dict] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored dirs in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]

        rel_dir = str(Path(dirpath).relative_to(root))
        if rel_dir == ".":
            rel_dir = "(root)"

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix in IGNORE_EXTS:
                continue

            rel_path = str(fpath.relative_to(root))
            size = fpath.stat().st_size
            lang = LANGUAGE_MAP.get(fpath.suffix, "Other")

            entry = {"path": rel_path, "size_bytes": size, "language": lang}

            if fpath.suffix == ".py":
                imports = extract_python_imports(fpath)
                if imports:
                    entry["imports"] = imports

            files.append(entry)

            if fname in ENTRY_NAMES:
                entry_points.append(rel_path)

            if rel_dir not in dir_summary:
                dir_summary[rel_dir] = {"file_count": 0, "total_bytes": 0, "languages": set()}
            dir_summary[rel_dir]["file_count"] += 1
            dir_summary[rel_dir]["total_bytes"] += size
            dir_summary[rel_dir]["languages"].add(lang)

    # Sort files by size descending — big files = important logic
    files.sort(key=lambda f: f["size_bytes"], reverse=True)

    # Top 50 files by size (enough signal for Claude)
    key_files = files[:50]

    # Serialise dir summary (sets → lists)
    dir_serialisable = {
        d: {
            "file_count": v["file_count"],
            "total_bytes": v["total_bytes"],
            "languages": sorted(v["languages"]),
        }
        for d, v in sorted(dir_summary.items())
    }

    return {
        "repo_root": str(root),
        "total_files": len(files),
        "entry_points": entry_points,
        "top_level_directories": dir_serialisable,
        "key_files": key_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a structured repo map for Claude.")
    parser.add_argument("--path", default=".", help="Path to the repository root")
    parser.add_argument("--output", default=None, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    repo_map = walk_repo(Path(args.path))

    output = json.dumps(repo_map, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Repo map written to {args.output} ({repo_map['total_files']} files)")
    else:
        print(output)


if __name__ == "__main__":
    main()
