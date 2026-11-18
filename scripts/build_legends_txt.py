#!/usr/bin/env python3
"""Generate figures/legends.txt from FIGURE_LEGENDS.md.

FIGURE_LEGENDS.md is the single source of truth for figure legends. This script
renders it into the key/TITLE/LEGEND/END format that generate_presentation.py
parses, so the two never drift apart again.

Keys must match how generate_presentation.py resolves image paths:

    Figure<N>      -> figures/Figure<N>/Figure<N>.png
    Appendix       -> figures/Appendix/FigureAppendix.png     (Appendix 1)
    Appendix<n>    -> figures/Appendix/FigureAppendix<n>.png
    Validation     -> figures/Validation/FigureValidation.png

Run:  python3 scripts/build_legends_txt.py
"""

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SRC = os.path.join(ROOT, "FIGURE_LEGENDS.md")
DST = os.path.join(ROOT, "figures", "legends.txt")

# Sections of FIGURE_LEGENDS.md that are not figures.
SKIP = ("References cited in the legends",)


def heading_to_key(name):
    """Map a FIGURE_LEGENDS.md heading onto a generate_presentation.py key."""
    m = re.match(r"Figure\s+(\d+)$", name)
    if m:
        return "Figure" + m.group(1)
    m = re.match(r"Appendix\s+(\d+)$", name)
    if m:
        # Appendix 1 is stored unnumbered as FigureAppendix.png.
        return "Appendix" if m.group(1) == "1" else "Appendix" + m.group(1)
    if name.lower().startswith("validation"):
        return "Validation"
    return None


def demarkdown(text):
    """Flatten the markdown that would otherwise render literally on a slide."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # bold
    text = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"\1", text)  # italic
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"`([^`]+)`", r"\1", text)              # code spans
    return text


def parse(md):
    """Yield (key, title, [body lines]) for each figure section, in file order."""
    sections, cur = [], None
    for line in md.splitlines():
        if line.startswith("## "):
            if cur:
                sections.append(cur)
            head = line[3:].strip()
            # "Figure 3 — Recording length determines ..." -> name, title
            parts = re.split(r"\s+[—–-]\s+", head, maxsplit=1)
            name = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else name
            cur = {"name": name, "title": title, "body": []}
        elif cur is not None:
            if line.strip() in ("---", "***"):
                continue
            cur["body"].append(line)
    if cur:
        sections.append(cur)

    for s in sections:
        if s["name"] in SKIP:
            continue
        key = heading_to_key(s["name"])
        if key is None:
            print(f"  ! unmapped heading, skipped: {s['name']}", file=sys.stderr)
            continue
        yield key, s["name"], s["title"], s["body"]


def main():
    with open(SRC, encoding="utf-8") as f:
        md = f.read()

    out = [
        "# GENERATED FILE - do not edit by hand.",
        "# Source: FIGURE_LEGENDS.md   Rebuild: python3 scripts/build_legends_txt.py",
        "# Consumed by scripts/generate_presentation.py.",
        "",
    ]

    n = 0
    for key, name, title, body in parse(md):
        text = demarkdown("\n".join(body)).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        out += [f"### {key}", f"TITLE: {name}: {demarkdown(title)}", "LEGEND:"]
        out += [ln for ln in text.splitlines()]
        out += ["END", ""]
        n += 1

        img = (os.path.join(ROOT, "figures", "Appendix",
                            f"FigureAppendix{key.replace('Appendix', '')}.png")
               if key.startswith("Appendix")
               else os.path.join(ROOT, "figures", key, f"{key}.png"))
        if not os.path.exists(img):
            print(f"  ! {key}: no image at {os.path.relpath(img, ROOT)}", file=sys.stderr)

    with open(DST, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"  wrote {n} legends -> {os.path.relpath(DST, ROOT)}")


if __name__ == "__main__":
    main()
