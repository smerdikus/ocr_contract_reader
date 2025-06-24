#!/usr/bin/env python3
"""
read_from_pdf_smart.py  –  robustní extraktor kupních smluv v češtině

• Přednostně použije textovou vrstvu PDF (Chrome-style copy);
  OCR jen u obrázkových stránek nebo s --force-ocr.
• Fuzzy nadpisy (§1, §3) ±2 chyby.
• Parcely: token-by-token, dědění LV, všechny podíly v poli.
• Prodávající: všichni jmenovaní mezi „Smluvní strany“ a „Kupující“.
• Výstup JSON s klíči: smlouva, celkem cena, parcely, smluvni strany, validation.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import fitz        # PyMuPDF
import regex       # fuzzy regex
import numpy as np
import easyocr

warnings.filterwarnings("ignore", category=UserWarning)


# 1. Helpers: remove diacritics & fuzzy match
DIAC_MAP = str.maketrans(
    "áÁčČďĎéÉěĚíÍňŇóÓřŘšŠťŤúÚůŮýÝžŽ",
    "aAcCdDeEeEiInNoOrRsStTuUuUyYzZ",
)

def normalize_czech_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text.translate(DIAC_MAP).strip())

def _find_fuzzy(pattern: str, text: str, max_edits: int = 2):
    return regex.search(fr"({pattern}){{e<={max_edits}}}", text,
                        regex.I | regex.BESTMATCH)


# 2. Text extraction: layer first, OCR fallback
def _extract_text_layer(pdf: Path) -> List[str]:
    doc = fitz.open(pdf)
    pages = [p.get_text("text", sort=True) for p in doc]
    doc.close()
    return pages

def _ocr_pages(pdf: Path, lang: str = "cs") -> List[str]:
    reader = easyocr.Reader([lang])
    doc = fitz.open(pdf)
    results: List[str] = []
    for pg in doc:
        pix = pg.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            alpha = img[..., 3:] / 255.0
            img = (img[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        results.append(" ".join(seg[1] for seg in reader.readtext(img)))
    doc.close()
    return results


# 3. Parcels & shares extractor
def extract_parcels_and_shares(
    section: str
) -> Tuple[List[Dict[str, object]], List[List[str]]]:
    """
    Return unique parcels (cislo parcely, LV, podily:list) and
    parallel list of share-lists.
    """
    text = normalize_czech_text(section.lower())

    token_rx = regex.finditer(
        r"parc\s*[:\.]?\s*,?\s*(?:[cč]\s*[:\.]?)?\s*([0-9]+/[0-9]+)"
        r"|lv\s*[:\.]?\s*(?:[cč]\s*[:\.]?)?\s*(\d{1,6})"
        r"|(\d+/\d+)",
        text, regex.I,
    )

    parcels: List[Dict[str, object]] = []
    pending_lv: List[Dict[str, object]] = []
    last_rec: Dict[str, object] | None = None

    for m in token_rx:
        parc, lv, frac = m.group(1), m.group(2), m.group(3)
        if parc:
            rec = {"cislo parcely": parc, "LV": "", "podily": []}
            parcels.append(rec)
            pending_lv.append(rec)
            last_rec = rec
        elif lv and pending_lv:
            for r in pending_lv:
                r["LV"] = lv
            pending_lv.clear()
        elif frac and last_rec is not None:
            last_rec["podily"].append(frac)

    # dedupe by (parcel, LV, tuple(podily))
    unique, seen = [], set()
    for p in parcels:
        key = (p["cislo parcely"], p["LV"], tuple(p["podily"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    shares = [p["podily"] for p in unique]
    return unique, shares


# 4. Sellers extractor
def extract_sellers_between(header: str) -> List[Tuple[str, str]]:
    """
    Extract sellers between fuzzy "Smluvní strany" and "Kupující".
    Pattern: Name Surname, RC ###/###, bytem ...
    """
    hdr_norm = normalize_czech_text(header.lower())
    start_m = _find_fuzzy(r"smluvni\s+strany", hdr_norm)
    start = start_m.end() if start_m else 0
    end_m = _find_fuzzy(r"kupujici", hdr_norm)
    end = end_m.start() if end_m else len(header)

    segment = normalize_czech_text(header[start:end])
    pat = regex.compile(
        r"(?<!\S)"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
        r"\s*,\s*rc\s*[\d/]+\s*,\s*bytem\s+([^,;\)]+)",
        regex.I,
    )
    return [(m.group(1).strip(), m.group(2).strip())
            for m in pat.finditer(segment)]


# 5. Split into header, §1, §3
def split_sections(full: str) -> Dict[str, str]:
    comp = " ".join(full.splitlines())
    base = normalize_czech_text(comp.lower())
    m1 = _find_fuzzy(r"uvodni\s+ustanoveni", base)
    m3 = _find_fuzzy(r"kupni\s+cena", base)
    m4 = _find_fuzzy(r"prohlaseni", base)

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


# 6. Build final JSON template
def build_template(parsed: Dict[str, str]) -> Dict:
    header = normalize_czech_text(parsed["header"])
    sec1   = parsed["s1"]
    full   = normalize_czech_text(parsed["full_text"])

    tpl: Dict = {
        "smlouva": "",
        "celkem cena": "",
        "parcely": [],
        "smluvni strany": [],
        "validation": {"price_sum_matches": False,
                       "calculated_sum": "0",
                       "difference": "0"},
    }

    # smlouva
    m = re.search(r"cislo:\s*([A-Z0-9]+)", header, re.I)
    if m:
        tpl["smlouva"] = m.group(1)

    # ceny
    prices = [int(p.replace(" ", ""))
              for p in re.findall(r"(\d+[\s\d]*)\s*kc", full, re.I)]
    total = prices[0] if prices else 0
    tpl["celkem cena"] = str(total)

    # prodávající
    sellers = extract_sellers_between(header) or [("", "")]
    parcels, shares = extract_parcels_and_shares(sec1)
    tpl["parcely"] = parcels

    for i, (name, addr) in enumerate(sellers):
        price = (prices[i+1] if i+1 < len(prices)
                 else total if i == 0 else 0)
        tpl["smluvni strany"].append({
            "jmeno": name,
            "bydliste": addr,
            "cena": str(price),
        })

    # validace
    sum_i = sum(int(s["cena"]) for s in tpl["smluvni strany"])
    tpl["validation"].update({
        "calculated_sum": str(sum_i),
        "difference": str(abs(total - sum_i)),
        "price_sum_matches": total == sum_i,
    })

    return tpl


# 7. Pipeline
def process(pdf: Path, *, force_ocr: bool = False):
    pages = _extract_text_layer(pdf) if not force_ocr else []
    if not any(p.strip() for p in pages):
        print("» OCR fallback …")
        pages = _ocr_pages(pdf)
    else:
        print("» Embedded text layer used.")

    full_text = "\n".join(pages)
    secs = split_sections(full_text)
    parsed = {**secs, "full_text": full_text.strip(), "total_characters": len(full_text)}
    clean  = {k: normalize_czech_text(v) for k, v in parsed.items()}
    tpl    = build_template(clean)

    out_dir = pdf.with_suffix("").name + "_out"
    os.makedirs(out_dir, exist_ok=True)

    with open(Path(out_dir) / "parsed_contract.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    with open(Path(out_dir) / "template.json", "w", encoding="utf-8") as f:
        json.dump(tpl, f, ensure_ascii=False, indent=2)

    print(f"OK: Výstup uložen do '{out_dir}/'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extraktor kupních smluv (PDF -> JSON)")
    ap.add_argument("pdf", type=Path, help="vstupní PDF")
    ap.add_argument("--force-ocr", action="store_true",
                    help="OCR i když je textová vrstva přítomna")
    args = ap.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"ERROR: soubor '{args.pdf}' nenalezen")

    process(args.pdf, force_ocr=args.force_ocr)
