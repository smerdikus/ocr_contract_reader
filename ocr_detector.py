#!/usr/bin/env python3
"""
read_from_pdf_smart.py  -  robustní extraktor kupních smluv v češtině

•  Přednostně použije textovou vrstvu PDF (stejnou, jakou umí kopírovat Chrome);
   k OCR sahá jen u stránek-obrázků nebo s přepínačem --force-ocr.
•  Fuzzy detekce nadpisů (ÚVODNÍ USTANOVENÍ, KUPNÍ CENA …) ±2 překlepy.
•  Parcely: procházení token-po-tokenu → každá „parc. č NNN/NN“ dostane zděděné
   LV a první zlomek za sebou jako podíl; deduplikace (parcelní č., LV).
•  Prodávající: hledají se jen v části hlavičky PŘED slovem „Kupující“ a řádek
   musí obsahovat „RC“, aby se nerozpoznali kupující nebo jiné velké řetězce.
•  Výstup JSON:

    {
      "smlouva": "...",
      "celkem cena": "...",
      "parcely": [
          {"cislo parcely": "...", "LV": "..."},
          …
      ],
      "smluvni strany": [
          {"jmeno": "...", "bydliste": "...", "parcely": ["1/54", …], "cena": "..."},
          …
      ],
      "validation": {...}
    }
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

import fitz                   # PyMuPDF
import regex                  # fuzzy regex
import numpy as np
import easyocr

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Diakritika & fuzzy helper
# ────────────────────────────────────────────────────────────────────────────────
def normalize_czech_text(text: str) -> str:
    """Odstraní české diakritiky, zkrátí bílé mezery."""
    if not isinstance(text, str):
        return text
    tbl = str.maketrans(
        "áÁčČďĎéÉěĚíÍňŇóÓřŘšŠťŤúÚůŮýÝžŽ",
        "aAcCdDeEeEiInNoOrRsStTuUuUyYzZ",
    )
    return re.sub(r"\s+", " ", text.translate(tbl).strip())


def _find_fuzzy(pattern: str, text: str, max_edits: int = 2):
    """Najde nejlepší shodu s ≤ max_edits Levenshteinovými úpravami."""
    return regex.search(fr"({pattern}){{e<={max_edits}}}", text,
                        regex.I | regex.BESTMATCH)

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Textové vrstvy / OCR
# ────────────────────────────────────────────────────────────────────────────────
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
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:                                    # RGBA → RGB (bílé pozadí)
            alpha = img[..., 3:] / 255.0
            img = (img[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        results.append(" ".join(seg[1] for seg in reader.readtext(img)))
    doc.close()
    return results

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Parcely & podíly
# ────────────────────────────────────────────────────────────────────────────────
def extract_parcels_and_shares(section: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Vrátí (unikátní parcely s LV) + list zlomků (podílů) v pořadí výskytu.
    LV se „dědí“ dozadu k předchozím parcelám v odstavci, než se objeví nové LV.
    """
    text = normalize_czech_text(section.lower())

    token_rx = regex.finditer(
        r"parc\s*\.?,?\s*c\s*([0-9]+/[0-9]+)"   # 1: parcela
        r"|lv\s*c?\.?\s*(\d{1,6})"              # 2: LV
        r"|(\d+/\d+)",                          # 3: zlomek
        text,
        regex.I,
    )

    parcels, pending = [], []                  # pending čeká na LV
    shares, share_seen = [], set()

    def push_share(frac: str):
        if frac not in share_seen:
            share_seen.add(frac)
            shares.append(frac)

    for tok in token_rx:
        parc, lv, frac = tok.group(1), tok.group(2), tok.group(3)
        if parc:
            rec = {"cislo parcely": parc, "LV": ""}
            parcels.append(rec)
            pending.append(rec)
        elif lv:
            for rec in pending:
                rec["LV"] = lv
            pending.clear()
        elif frac:
            push_share(frac)

    # deduplikace podle (parcela, LV)
    unique, seen_pairs = [], set()
    for p in parcels:
        key = (p["cislo parcely"], p["LV"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique.append(p)

    return unique, shares

# ────────────────────────────────────────────────────────────────────────────────
# 4.  Rozdělení na hlavičku / §1 / §3
# ────────────────────────────────────────────────────────────────────────────────
def split_sections(full: str) -> Dict[str, str]:
    compact = " ".join(full.splitlines())
    base    = normalize_czech_text(compact.lower())

    m1 = _find_fuzzy(r"uvodni\s+ustanoveni", base)
    m3 = _find_fuzzy(r"kupni\s+cena", base)
    m4 = _find_fuzzy(r"prohlaseni", base)

    out = {"header": "", "s1": "", "s3": ""}
    if m1:
        out["header"] = compact[: m1.start()].strip()
        stop1 = (m3 or m4 or regex.Match("", pos=len(compact))).start()
        out["s1"] = compact[m1.start(): stop1].strip()
    if m3:
        stop3 = (m4 or regex.Match("", pos=len(compact))).start()
        out["s3"] = compact[m3.start(): stop3].strip()

    for k in out:
        out[k] = re.sub(r"\s+", " ", out[k]).strip()
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 5.  Sestavení finální šablony
# ────────────────────────────────────────────────────────────────────────────────
def build_template(parsed: Dict) -> Dict:
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

    # číslo smlouvy
    m = re.search(r"cislo:\s*([A-Z0-9]+)", header, re.I)
    if m:
        tpl["smlouva"] = m.group(1)

    # ceny
    prices = [int(p.replace(" ", "")) for p in
              re.findall(r"(\d+[\s\d]*)\s*kc", full, re.I)]
    total = prices[0] if prices else 0
    tpl["celkem cena"] = str(total)

    # prodávající (před slovem Kupujici + musí obsahovat RC)
    cut = header.lower().find("kupujici")
    hdr_slice = header[:cut] if cut != -1 else header
    sellers = re.findall(
        r"([A-Z][a-z]+\s+[A-Z][a-z]+),\s*rc\s*[\d/]+,\s*bytem\s+([^,;\)]+)",
        hdr_slice, re.I)
    if not sellers:
        sellers = [("", "")]                              # placeholder

    # parcely & podíly
    parcels, shares = extract_parcels_and_shares(sec1)
    tpl["parcely"] = parcels

    for i, (name, addr) in enumerate(sellers):
        tpl["smluvni strany"].append({
            "jmeno": name.strip(),
            "bydliste": addr.strip(),
            "parcely": shares.copy(),
            "cena": str(prices[i+1] if i+1 < len(prices)
                        else total if i == 0 else 0),
        })

    sum_indiv = sum(int(s["cena"]) for s in tpl["smluvni strany"])
    tpl["validation"].update({
        "calculated_sum": str(sum_indiv),
        "difference": str(abs(total - sum_indiv)),
        "price_sum_matches": total == sum_indiv,
    })
    return tpl

# ────────────────────────────────────────────────────────────────────────────────
# 6.  Pipeline
# ────────────────────────────────────────────────────────────────────────────────
def process(pdf: Path, *, force_ocr: bool = False):
    pages = _extract_text_layer(pdf) if not force_ocr else []
    if not any(p.strip() for p in pages):
        print("» OCR fallback …")
        pages = _ocr_pages(pdf)
    else:
        print("» Embedded text layer used.")

    full_text = "\n".join(pages)
    parts     = split_sections(full_text)

    parsed = {
        **parts,
        "full_text": full_text.strip(),
        "total_characters": len(full_text),
    }
    clean = {k: normalize_czech_text(v) if isinstance(v, str) else v
             for k, v in parsed.items()}

    template = build_template(clean)

    out_dir = pdf.with_suffix("").name + "_out"
    os.makedirs(out_dir, exist_ok=True)
    with open(Path(out_dir) / "parsed_contract.json", "w",
              encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    with open(Path(out_dir) / "template.json", "w",
              encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"✔ Výstup uložen do '{out_dir}/'")

# ────────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Extraktor dat z kupních smluv (PDF → JSON)")
    ap.add_argument("pdf", type=Path, help="vstupní PDF")
    ap.add_argument("--force-ocr", action="store_true",
                    help="ignorovat textovou vrstvu a OCR-ovat vše")
    args = ap.parse_args()

    if not args.pdf.is_file():
        sys.stderr.write(f"ERROR: soubor '{args.pdf}' nelze najít\n")
        sys.exit(1)

    process(args.pdf, force_ocr=args.force_ocr)


if __name__ == "__main__":
    main()
