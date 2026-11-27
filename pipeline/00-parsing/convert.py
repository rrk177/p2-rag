import re
from pathlib import Path

import fitz                      
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "md"
FOOTER_PT, HEADER_PT = 50, 40


def md_table(rows):
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def pdf_to_md(path):
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            tabs = page.find_tables().tables
            boxes = [fitz.Rect(t.bbox) for t in tabs]
            items = [(t.bbox[1], md_table(t.extract())) for t in tabs]
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0 or any(fitz.Rect(b["bbox"]).intersects(r) for r in boxes):
                    continue
                text = " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"]).strip()
                if not text or b["bbox"][1] > page.rect.height - FOOTER_PT or b["bbox"][3] < HEADER_PT:
                    continue
                size = max(s["size"] for l in b["lines"] for s in l["spans"])
                if size >= 15:
                    text = "# " + text
                elif size >= 12.5:
                    text = "## " + text
                elif text.startswith("• "):
                    text = "- " + text[2:]
                items.append((b["bbox"][1], text))
            out += [t for _, t in sorted(items, key=lambda x: x[0])]
    return "\n\n".join(out)


def docx_to_md(path):
    doc = Document(path)
    out, n = [], 0
    for el in doc.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "tbl":
            out.append(md_table([[c.text for c in r.cells] for r in Table(el, doc).rows]))
            continue
        if tag != "p":
            continue
        p = Paragraph(el, doc)
        text, style = p.text.strip(), p.style.name
        if not text:
            continue
        if style == "List Number":
            n += 1
            out.append(f"{n}. {text}")
            continue
        n = 0
        prefix = {"Heading 1": "# ", "Heading 2": "## ", "List Bullet": "- "}.get(style, "")
        out.append(prefix + text)
    return "\n\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".pdf", ".docx"))
    for p in files:
        md = pdf_to_md(p) if p.suffix.lower() == ".pdf" else docx_to_md(p)
        md = re.sub(r"[ \t]+", " ", md).strip() + "\n"
        dst = OUT / (p.stem + ".md")
        dst.write_text(md, encoding="utf-8")
        print(f"[parse] {p.name:44} -> data/md/{dst.name} ({len(md)} симв.)")
    print(f"[parse] готово: {len(files)} документов")


if __name__ == "__main__":
    main()
