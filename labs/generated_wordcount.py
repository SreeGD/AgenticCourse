"""
wordcount.py — CLI Word Frequency Counter

Reads a UTF-8 plain-text file, tokenizes its contents into words, counts
occurrences, and prints a ranked table of the top-N most frequent words to
stdout.

Usage:
    python wordcount.py <file> [--top N]
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Generator, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    """Argparse type validator: must be a positive integer (≥ 1)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid integer."
        )
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"--top must be a positive integer (≥ 1), got {ivalue}."
        )
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    """Return a configured ArgumentParser for wordcount."""
    parser = argparse.ArgumentParser(
        prog="wordcount.py",
        description=(
            "Count word frequencies in a UTF-8 plain-text file and print "
            "a ranked table of the most frequent words."
        ),
    )
    parser.add_argument(
        "file",
        metavar="FILE",
        help="Path to the UTF-8 plain-text input file.",
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=_positive_int,
        default=10,
        help="Number of top words to display (default: 10, must be ≥ 1).",
    )
    return parser


# ---------------------------------------------------------------------------
# UTF-8 file reader (streaming, line-by-line)
# ---------------------------------------------------------------------------

def read_lines(file_path: str) -> Generator[str, None, None]:
    """
    Open *file_path* with UTF-8 encoding and yield lines one at a time.

    Raises SystemExit(1) with a human-readable stderr message on:
      - FileNotFoundError
      - PermissionError
      - UnicodeDecodeError
      - Any other OSError
    """
    path = Path(file_path)
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                yield line
    except FileNotFoundError:
        print(
            f"Error: File not found: '{file_path}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except PermissionError:
        print(
            f"Error: Permission denied when reading '{file_path}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except UnicodeDecodeError as exc:
        print(
            f"Error: '{file_path}' contains non-UTF-8 bytes and cannot be "
            f"decoded. ({exc})",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except OSError as exc:
        print(
            f"Error: Could not read '{file_path}': {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Tokenization and normalization
# ---------------------------------------------------------------------------

# Strip any leading/trailing characters that are NOT alphanumeric or apostrophe.
_STRIP_PATTERN = re.compile(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$")

# Match tokens that consist solely of digits (after stripping).
_DIGITS_ONLY = re.compile(r"^\d+$")


def tokenize(lines: Iterable[str]) -> Generator[str, None, None]:
    """
    Yield cleaned, lowercase tokens from an iterable of text lines.

    Pipeline per token:
      1. Split line on whitespace.
      2. Strip leading/trailing punctuation (anything not in [a-zA-Z0-9']).
      3. Lowercase.
      4. Discard empty strings.
      5. Discard purely-numeric tokens.
    """
    for line in lines:
        for raw_token in line.split():
            token = _STRIP_PATTERN.sub("", raw_token).lower()
            if not token:
                continue
            if _DIGITS_ONLY.match(token):
                continue
            yield token


# ---------------------------------------------------------------------------
# Word frequency counting and ranking
# ---------------------------------------------------------------------------

def count_and_rank(
    tokens: Iterable[str], top_n: int
) -> List[Tuple[int, str, int]]:
    """
    Count token frequencies and return the top-*top_n* entries.

    Sorting rules:
      - Primary: descending by count.
      - Secondary: ascending alphabetically (for deterministic tie-breaking).

    Returns a list of (rank, word, count) tuples.
    If *top_n* exceeds the number of unique words, all unique words are returned.
    """
    counter: Counter = Counter(tokens)
    if not counter:
        return []

    # Sort: descending count, then ascending word for ties.
    sorted_words = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    top_entries = sorted_words[:top_n]

    return [(rank, word, count) for rank, (word, count) in enumerate(top_entries, start=1)]


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------

# Column widths as per spec.
_RANK_WIDTH = 4
_WORD_WIDTH = 20
_COUNT_WIDTH = 8

# Header and separator built once.
_HEADER = (
    f"{'Rank':>{_RANK_WIDTH}} | {'Word':<{_WORD_WIDTH}} | {'Count':>{_COUNT_WIDTH}}"
)
_SEPARATOR = "-" * len(_HEADER)


def print_table(ranked: List[Tuple[int, str, int]]) -> None:
    """
    Print the ranked word-frequency table to stdout.

    If *ranked* is empty, prints 'No words found.' instead.
    """
    if not ranked:
        print("No words found.")
        return

    print(_HEADER)
    print(_SEPARATOR)
    for rank, word, count in ranked:
        print(
            f"{rank:>{_RANK_WIDTH}} | {word:<{_WORD_WIDTH}} | {count:>{_COUNT_WIDTH}}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, process the file, and print the frequency table."""
    parser = build_parser()
    args = parser.parse_args()

    # Stream lines from the file (exits with code 1 on I/O errors).
    lines = read_lines(args.file)

    # Tokenize and normalize.
    tokens = tokenize(lines)

    # Count and rank.
    ranked = count_and_rank(tokens, args.top)

    # Print results.
    print_table(ranked)


if __name__ == "__main__":
    main()
