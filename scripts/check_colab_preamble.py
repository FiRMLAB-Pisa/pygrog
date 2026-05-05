from pathlib import Path
import sys


def main() -> int:
    conf_path = Path(__file__).resolve().parents[1] / "docs" / "conf.py"
    text = conf_path.read_text(encoding="utf-8")

    required_snippets = [
        '"first_notebook_cell"',
        "google.colab",
        "pip",
        "pygrog",
    ]

    missing = [s for s in required_snippets if s not in text]
    if missing:
        print("Missing Colab preamble markers in docs/conf.py:")
        for item in missing:
            print(f"  - {item}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
