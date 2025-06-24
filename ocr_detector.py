#!/usr/bin/env python3
"""
read_from_pdf_smart.py – robustní extraktor kupních smluv v češtině
+ porovnání s referenční databází (XLSX)

Všechy výstupy (JSON, CSV, diff) ukládá do složky `output/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import fitz       # PyMuPDF
import numpy as np
import pandas as pd
import regex      # fuzzy regex
import easyocr

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def normalize_czech_text(text: str) -> str:
    """Odstraní diakritiku a sjednotí bílé znaky."""
    DIAC_MAP = str.maketrans(
        "áÁčČďĎéÉěĚíÍňŇóÓřŘšŠťŤúÚůŮýÝžŽ",
        "aAcCdDeEeEiInNoOrRsStTuUuUyYzZ",
    )
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text.translate(DIAC_MAP).strip())


def fuzzy_search(pattern: str, text: str, max_edits: int = 2):
    """Fuzzy regex vyhledávání s povolením `max_edits` chyb."""
    return regex.search(fr"({pattern}){{e<={max_edits}}}", text,
                        regex.I | regex.BESTMATCH)


# ──────────────────────────────────────────────────────────────────────────────
# Text Extraction (PDF layer / OCR)
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_layer(pdf: Path) -> List[str]:
    doc = fitz.open(pdf)
    pages = [page.get_text("text", sort=True) for page in doc]
    doc.close()
    return pages


def ocr_pages(pdf: Path, lang: str = "cs") -> List[str]:
    reader = easyocr.Reader([lang])
    doc = fitz.open(pdf)
    results: List[str] = []

    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # odstranění alfa kanálu
        if pix.n == 4:
            alpha = img[..., 3:] / 255.0
            img = (img[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)

        text = " ".join(seg[1] for seg in reader.readtext(img))
        results.append(text)

    doc.close()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Parcel & Share Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_parcels_and_shares(
    section: str
) -> Tuple[List[Dict[str, object]], List[List[str]]]:
    """Najde čísla parcel, LV a podíly v textové sekci."""
    text = normalize_czech_text(section.lower())
    token_rx = regex.finditer(
        r"parc\s*[:\.]?\s*,?\s*(?:[cc]\s*[:\.]?)?\s*([0-9]+/[0-9]+)"
        r"|lv\s*[:\.]?\s*(?:[cc]\s*[:\.]?)?\s*(\d{1,6})"
        r"|(\d+/\d+)",
        text,
        regex.I
    )

    parcels: List[Dict[str, object]] = []
    pending_lv: List[Dict[str, object]] = []
    last_rec = None

    for m in token_rx:
        parc, lv, frac = m.group(1), m.group(2), m.group(3)

        if parc:
            rec = {"cislo_parcely": parc, "lv": "", "podily": []}
            parcels.append(rec)
            pending_lv.append(rec)
            last_rec = rec

        elif lv and pending_lv:
            for r in pending_lv:
                r["lv"] = lv
            pending_lv.clear()

        elif frac and last_rec is not None:
            last_rec["podily"].append(frac)

    # deduplikace
    unique, seen = [], set()
    for p in parcels:
        key = (p["cislo_parcely"], p["lv"], tuple(p["podily"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    shares = [p["podily"] for p in unique]
    return unique, shares


# ──────────────────────────────────────────────────────────────────────────────
# Seller Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_sellers_between(header: str) -> List[Tuple[str, str]]:
    """Vybere prodávající mezi 'Smluvní strany' a 'Kupující'."""
    hdr = normalize_czech_text(header.lower())
    start_m = fuzzy_search(r"smluvni\s+strany", hdr)
    start = start_m.end() if start_m else 0
    end_m = fuzzy_search(r"kupujici", hdr)
    end = end_m.start() if end_m else len(header)

    segment = normalize_czech_text(header[start:end])
    seller_rx = regex.compile(
        r"(?<!\S)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
        r"\s*,\s*rc\s*[\d/]+\s*,\s*bytem\s+(.+?)"
        r"(?=\s*(?:\(dale\s+jen|;|$))",
        regex.I | regex.S
    )

    return [(m.group(1).strip(), m.group(2).strip())
            for m in seller_rx.finditer(segment)]


# ──────────────────────────────────────────────────────────────────────────────
# Section Splitting
# ──────────────────────────────────────────────────────────────────────────────

def split_sections(full: str) -> Dict[str, str]:
    """Rozdělí celý text na header, §1 a §3."""
    comp = " ".join(full.splitlines())
    base = normalize_czech_text(comp.lower())
    m1 = fuzzy_search(r"uvodni\s+ustanoveni", base)
    m3 = fuzzy_search(r"kupni\s+cena", base)
    m4 = fuzzy_search(r"prohlaseni", base)

    out = {"header": "", "s1": "", "s3": ""}

    if m1:
        out["header"] = comp[:m1.start()].strip()
        stop1 = (m3 or m4 or regex.Match("", pos=len(comp))).start()
        out["s1"] = comp[m1.start():stop1].strip()

    if m3:
        stop3 = (m4 or regex.Match("", pos=len(comp))).start()
        out["s3"] = comp[m3.start():stop3].strip()

    for k in out:
        out[k] = re.sub(r"\s+", " ", out[k]).strip()

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Template Builder & CSV Export
# ──────────────────────────────────────────────────────────────────────────────

def build_template(parsed: Dict[str, str]) -> Dict:
    """Sestaví výslednou JSON šablonu s validací cen."""
    header = normalize_czech_text(parsed["header"])
    sec1 = parsed["s1"]
    full = normalize_czech_text(parsed["full_text"])

    tpl = {
        "smlouva": "",
        "celkem_cena": "",
        "parcely": [],
        "smluvni_strany": [],
        "validation": {
            "price_sum_matches": False,
            "calculated_sum": "0",
            "difference": "0",
        },
    }

    if m := re.search(r"cislo:\s*([A-Z0-9]+)", header, re.I):
        tpl["smlouva"] = m.group(1)

    prices = [int(p.replace(" ", "")) for p in re.findall(r"(\d+[\s\d]*)\s*kc", full, re.I)]
    total = prices[0] if prices else 0
    tpl["celkem_cena"] = str(total)

    sellers = extract_sellers_between(header) or [("", "")]
    parcels, shares = extract_parcels_and_shares(sec1)
    tpl["parcely"] = parcels

    for i, (name, addr) in enumerate(sellers):
        price = prices[i + 1] if i + 1 < len(prices) else (total if i == 0 else 0)
        tpl["smluvni_strany"].append({
            "jmeno": name,
            "bydliste": addr,
            "cena": str(price),
        })

    sum_i = sum(int(s["cena"]) for s in tpl["smluvni_strany"])
    tpl["validation"].update({
        "calculated_sum": str(sum_i),
        "difference": str(abs(total - sum_i)),
        "price_sum_matches": total == sum_i,
    })

    return tpl


def save_csv_from_template(template: Dict, out_dir: Path) -> Path:
    """Vygeneruje CSV z šablony a vrátí jeho cestu."""
    csv_path = out_dir / "template.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["owner_fullname", "owner_address", "price",
                         "plat_number", "lv_number", "ratio"])

        parcels = template["parcely"]
        owners = template["smluvni_strany"]

        for idx, owner in enumerate(owners):
            for parc in parcels:
                ratio = parc["podily"][idx] if idx < len(parc["podily"]) else ""
                writer.writerow([
                    owner["jmeno"],
                    owner["bydliste"],
                    owner["cena"],
                    parc["cislo_parcely"],
                    parc["lv"],
                    ratio,
                ])

    print(f"✔ CSV uloženo do '{csv_path}'")
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process(
    pdf: Path,
    *,
    pages_limit: int | None = None,
    force_ocr: bool = False,
) -> Path:
    """Spustí extrakci, OCR (volitelně), staví templáty i CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    pages = extract_text_layer(pdf) if not force_ocr else []
    if not any(p.strip() for p in pages):
        print(">> OCR fallback …")
        pages = ocr_pages(pdf)
    else:
        print(">> Embedded text layer used.")

    if pages_limit is not None:
        print(f">> Zpracovávám prvních {pages_limit} stran.")
        pages = pages[:pages_limit]

    full_text = "\n".join(pages)
    parsed = {**split_sections(full_text),
              "full_text": full_text.strip(),
              "total_characters": len(full_text)}
    clean = {k: normalize_czech_text(v) for k, v in parsed.items()}
    tpl = build_template(clean)

    # Uložení JSONů
    with open(OUTPUT_DIR / "parsed_contract.json", "w",
              encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "template.json", "w",
              encoding="utf-8") as f:
        json.dump(tpl, f, ensure_ascii=False, indent=2)

    # Uložení CSV
    return save_csv_from_template(tpl, OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison (sorted by values, zachovává pořadí sloupců)
# ──────────────────────────────────────────────────────────────────────────────

def compare_from_db_sorted(
    contract_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    diff_path: Path | None = None,
) -> None:
    """Porovná seřazené řádky CSV vs XLSX; ukládá diff do output/row_diff.csv."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) příprava a čištění
    c = (contract_df.astype(str)
         .fillna("")
         .applymap(str.strip))
    r = (ref_df.astype(str)
         .fillna("")
         .applymap(str.strip))

    c.columns = [col.lower().strip() for col in c.columns]
    r.columns = [col.lower().strip() for col in r.columns]

    # 2) zachovat pořadí sloupců z CSV, doplnit extras na konec
    orig_cols = list(c.columns)
    extra_cols = [col for col in r.columns if col not in orig_cols]
    all_cols = orig_cols + extra_cols

    c = c.reindex(columns=all_cols, fill_value="")
    r = r.reindex(columns=all_cols, fill_value="")

    # 3) seřazení podle hodnot
    c_sorted = c.sort_values(by=all_cols).reset_index(drop=True)
    r_sorted = r.sort_values(by=all_cols).reset_index(drop=True)

    # 4) porovnání řádek
    max_len = max(len(c_sorted), len(r_sorted))
    diffs: List[Dict[str, str]] = []

    for i in range(max_len):
        row_c = (c_sorted.iloc[i]
                 if i < len(c_sorted)
                 else pd.Series("", index=all_cols))
        row_r = (r_sorted.iloc[i]
                 if i < len(r_sorted)
                 else pd.Series("", index=all_cols))

        if not row_c.equals(row_r):
            diffs.append({"dataset": "csv",  **row_c.to_dict()})
            diffs.append({"dataset": "xlsx", **row_r.to_dict()})

    if not diffs:
        print("✔ Porovnání: žádné rozdíly.")
        return

    diff_df = pd.DataFrame(diffs)
    dst = diff_path or (OUTPUT_DIR / "row_diff.csv")
    diff_df.to_csv(dst, index=False, encoding="utf-8")
    print(f"! Nalezeno {len(diffs)//2} nesouhlasných řádků – uložen report do '{dst}'.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraktor kupních smluv (PDF → CSV) + porovnání s XLSX"
    )
    parser.add_argument("pdf",  type=Path,
                        help="vstupní PDF")
    parser.add_argument("xlsx", type=Path,
                        help="vstupní XLSX pro porovnání")
    parser.add_argument("pages", type=int, nargs="?",
                        help="volitelný počet stran")
    parser.add_argument("--force-ocr", action="store_true",
                        help="OCR i když textová vrstva existuje")

    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"ERROR: soubor '{args.pdf}' nenalezen")
    if not args.xlsx.is_file():
        sys.exit(f"ERROR: soubor '{args.xlsx}' nenalezen")

    csv_path = process(
        args.pdf,
        pages_limit=args.pages,
        force_ocr=args.force_ocr
    )

    contract_df = pd.read_csv(csv_path, dtype=str)
    ref_df = pd.read_excel(args.xlsx, dtype=str)

    compare_from_db_sorted(contract_df, ref_df)

