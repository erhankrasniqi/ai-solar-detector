import os, io, re, time, json, argparse, pathlib, string
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
from PIL import Image, ImageDraw

# ==================== CONFIGURATION ====================
OPENAI_MODEL = "gpt-5"
EXCEL_PATH   = r"C:\path\to\address_list.xlsx"
SHEET_NAME   = None  # None -> first sheet
# ======================================================

# =====================  OpenAI Vision  ============================
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


def _to_data_url(img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _clean_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s.strip())
    return s


def detect_with_openai(img: Image.Image, model: str = OPENAI_MODEL) -> Tuple[bool, float, str]:
    """Returns (has_solar, confidence, raw_json_text)."""
    if not _OPENAI_OK:
        raise RuntimeError("OpenAI SDK not installed. Try: pip install openai")

    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    data_url = _to_data_url(img)

    system_msg = (
        "You are a strict JSON generator. Always reply with a single JSON object only, "
        "no prose, no backticks."
    )
    user_instr = (
        "Look at the aerial view of the building (focus on the roof). "
        "Are there visible solar panels? "
        "Return EXACTLY these fields as a single JSON:\n"
        '{ "has_solar": true/false, "confidence": number 0..1, "reason": "short text" }'
    )

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instr},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
    )

    content = resp.choices[0].message.content or "{}"
    content = _clean_json_text(content)

    try:
        out = json.loads(content)
    except Exception:
        out = {"has_solar": False, "confidence": 0.0, "reason": "parse_error", "raw": content}

    has_solar = bool(out.get("has_solar", False))
    conf = float(out.get("confidence", 0.0))
    return has_solar, conf, json.dumps(out, ensure_ascii=False)


# ============================  UTILITIES  ====================================
def clean_address_from_filename(fname: str) -> str:
    """Converts a filename into a readable address."""
    stem = pathlib.Path(fname).stem
    out = re.sub(r"[_]+", " ", stem)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


def parse_file_address(addr: str) -> Dict[str, str]:
    """Extracts structured address fields from a filename."""
    a = addr.replace(",", " ")
    m_street = re.split(r"\d", a, maxsplit=1)
    street_name = m_street[0].strip() if m_street else a.strip()
    m_house = re.search(r"\b(\d+)\b", a)
    house = m_house.group(1) if m_house else ""
    m_postal = re.search(r"\b(\d{4})\b", a)
    postal = m_postal.group(1) if m_postal else ""
    city = ""
    if postal:
        parts = re.split(r"\b"+re.escape(postal)+r"\b", a, maxsplit=1)
        if len(parts) > 1:
            tail = normalize_text(parts[1])
            city = tail.split(" ")[0] if tail else ""
    return {
        "street_name": normalize_text(street_name),
        "house": normalize_text(house),
        "postal": normalize_text(postal),
        "city": normalize_text(city),
        "all": normalize_text(addr),
    }


def guess_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for c in df.columns:
        cl = c.lower().strip()
        if any(re.search(p, cl, re.I) for p in patterns):
            return c
    return None


def ensure_output_cols(df: pd.DataFrame):
    for c in ["has", "confidence", "notes"]:
        if c not in df.columns:
            df[c] = ""


def score_match(file_info: Dict[str, str], row_vals: Dict[str, str]) -> int:
    """
    Returns a matching score:
      +3 if street matches
      +3 if house number matches
      +1 if postal code matches
      +1 if city matches
    """
    s = 0
    fs, rs = file_info["street_name"], row_vals.get("street", "")
    if fs and rs and (fs in rs or rs in fs):
        s += 3
    fh, rh = file_info["house"], row_vals.get("house", "")
    if fh and rh and fh == rh:
        s += 3
    fp, rp = file_info["postal"], row_vals.get("postal", "")
    if fp and rp and fp == rp:
        s += 1
    fc, rc = file_info["city"], row_vals.get("city", "")
    if fc and rc and fc == rc:
        s += 1
    return s


def save_overlay(img: Image.Image, label: str) -> Image.Image:
    """Adds a label at the bottom of an image."""
    copy = img.copy()
    d = ImageDraw.Draw(copy)
    w, h = copy.size
    box_h = 36
    d.rectangle([0, h - box_h, w, h], fill=(0, 0, 0, 140))
    d.text((10, h - box_h + 8), label, fill=(255, 255, 255))
    return copy


# ===============================  MAIN  ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="images", help="Folder with images (jpg/png)")
    ap.add_argument("--rate", type=float, default=1.0, help="Delay in seconds between requests")
    ap.add_argument("--save-overlays", action="store_true", help="Save labeled image copies")
    ap.add_argument("--overlay-dir", default="labeled", help="Output folder for labeled images")
    ap.add_argument("--excel-path", default=EXCEL_PATH, help="Existing Excel file to update")
    ap.add_argument("--sheet-name", default=SHEET_NAME, help="Sheet name (default: first)")
    args = ap.parse_args()

    img_dir = pathlib.Path(args.images_dir)
    if not img_dir.exists():
        raise SystemExit(f"Folder '{img_dir}' does not exist.")

    if args.save_overlays:
        pathlib.Path(args.overlay_dir).mkdir(parents=True, exist_ok=True)

    # Read existing Excel
    if not os.path.isfile(args.excel_path):
        raise SystemExit(f"Excel not found: {args.excel_path}")
    xls = pd.ExcelFile(args.excel_path)
    sheet = args.sheet_name or xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)

    # Find relevant columns
    addr_col = guess_column(df, [r"^address$", r"^adressa$", r"^adres", r"street"])
    house_col = guess_column(df, [r"house.*num", r"\bnumri", r"\bnr\b", r"house"])
    postal_col = guess_column(df, [r"postal", r"\bzip\b", r"postcode"])
    city_col   = guess_column(df, [r"\bcity\b", r"qytet", r"\btown\b"])

    if not addr_col:
        raise SystemExit("No address column found in Excel (e.g. address/adressa).")

    ensure_output_cols(df)

    # Add "image_path" column if missing
    if "image_path" not in df.columns:
        insert_at = df.columns.get_loc("notes") + 1 if "notes" in df.columns else len(df.columns)
        df.insert(insert_at, "image_path", "")

    # Prepare rows for comparison
    row_info: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        row_info.append({
            "street": normalize_text(r.get(addr_col, "")),
            "house":  normalize_text(str(r.get(house_col, ""))) if house_col else "",
            "postal": normalize_text(str(r.get(postal_col, ""))) if postal_col else "",
            "city":   normalize_text(str(r.get(city_col, ""))) if city_col else "",
        })

    files = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    files.sort()

    for i, p in enumerate(files, 1):
        try:
            img = Image.open(p).convert("RGB")
            addr_from_img = clean_address_from_filename(p.name)
            fi = parse_file_address(addr_from_img)

            has, conf, raw = detect_with_openai(img, OPENAI_MODEL)
            mark = "+" if has else "-"
            lbl = f"{mark}  conf={conf:.2f}"
            if args.save_overlays:
                save_overlay(img, lbl).save(pathlib.Path(args.overlay_dir) / p.name, "JPEG")

            best_idx, best_score = None, -1
            for idx, ri in enumerate(row_info):
                sc = score_match(fi, ri)
                if sc > best_score:
                    best_score, best_idx = sc, idx

            if best_idx is None:
                print(f"[{i}/{len(files)}] {p.name} -> no matching row found")
                continue

            df.at[best_idx, "has"] = mark
            df.at[best_idx, "confidence"] = round(float(conf), 3)
            df.at[best_idx, "notes"] = raw[:2000]
            df.at[best_idx, "image_path"] = p.name

            print(f"[{i}/{len(files)}] {p.name} -> row={best_idx+1} has={mark} conf={conf:.2f} score={best_score}")
            time.sleep(args.rate)

        except Exception as e:
            print(f"[{i}/{len(files)}] {p.name} -> ERROR: {e}")

    with pd.ExcelWriter(args.excel_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\nDone. Columns (has / confidence / notes / image_path) updated in: {args.excel_path} (sheet: {sheet})")
    if args.save_overlays:
        print(f"Labeled images saved to: {args.overlay_dir}")


if __name__ == "__main__":
    main()
