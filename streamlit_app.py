
import os, io, re, json
import numpy as np, cv2, streamlit as st, pandas as pd, easyocr
from pdf2image import convert_from_bytes
from app.config import ALLOWED_EXTS, LANG_LIST, DEFAULTS, MAX_FILE_SIZE_MB, PREVIEW_MAX_WIDTH, CITIES_HINT

st.set_page_config(page_title="OCR договор — усиленное извлечение v26", layout="wide")
st.title("📄 OCR: оригинал vs улучшение (v26)")

uploaded = st.file_uploader("Загрузите файл (PDF/PNG/JPG/TIFF)", type=[e.strip(".") for e in ALLOWED_EXTS])

def _render_image(img, caption=None):
    try:
        st.image(img, caption=caption)
    except:
        st.write(caption or "")
        st.image(img)

def _pdf_to_bgr_pages(data: bytes, dpi: int) -> list:
    pages = convert_from_bytes(data, fmt="png", dpi=dpi)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

def _decode_file_to_bgr_list(file_bytes: bytes, filename: str, dpi: int):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _pdf_to_bgr_pages(file_bytes, dpi)
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return [img] if img is not None else []

def _enhance(img_bgr: np.ndarray, scale: float, denoise_h: int, sharp_amount: float) -> np.ndarray:
    img = img_bgr.copy()
    if abs(scale - 1.0) > 1e-6:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    try:
        img = cv2.fastNlMeansDenoisingColored(img, None, h=denoise_h, hColor=denoise_h, templateWindowSize=7, searchWindowSize=21)
    except:
        pass
    try:
        blur = cv2.GaussianBlur(img, (0,0), 1.0)
        img = cv2.addWeighted(img, 1.0 + sharp_amount, blur, -sharp_amount, 0)
    except:
        pass
    return img

def _shrink(rgb_img: np.ndarray, max_w: int = PREVIEW_MAX_WIDTH) -> np.ndarray:
    h, w = rgb_img.shape[:2]
    if w <= max_w:
        return rgb_img
    scale = max_w / float(w)
    return cv2.resize(rgb_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# --- Text normalization & helpers ---
def _normalize_text(s: str) -> str:
    if not s:
        return ""
    LAT2CYR = str.maketrans({
        "A":"А","B":"В","E":"Е","K":"К","M":"М","H":"Н","O":"О","P":"Р","C":"С","T":"Т","X":"Х","Y":"У",
        "a":"а","e":"е","o":"о","p":"р","c":"с","x":"х","y":"у"
    })
    out = s.translate(LAT2CYR)
    out = re.sub(r"\bN[°ºoо]?\s*", "№ ", out, flags=re.IGNORECASE)
    out = out.replace("–","-").replace("—","-")
    out = re.sub(r"\s+", " ", out)
    return out.strip()

def _find_any_date(t: str):
    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/]((?:19|20)\d{2})\b", t)
    if m:
        dd, mm, yy = m.groups()
        return f"{dd.zfill(2)}.{mm.zfill(2)}.{yy}"
    # textual month (рус.)
    m = re.search(r'([«"]?)(\d{1,2})\1\s+([А-Яа-яA-Za-z]+)\s+((?:19|20)\d{2})', t)
    if m:
        d = m.group(2); mon = m.group(3).lower(); y = m.group(4)
        RU = {"январ":"01","феврал":"02","март":"03","апрел":"04","ма":"05","июн":"06","июл":"07","август":"08","сентябр":"09","октябр":"10","ноябр":"11","декабр":"12"}
        mm = None
        for k,v in RU.items():
            if mon.startswith(k): mm = v; break
        if mm:
            return f"{d.zfill(2)}.{mm}.{y}"
    return None

def _window_after_kw(t: str, kw_regex: str, max_len: int = 100):
    m = re.search(kw_regex + r"\s*[:\-]?\s*([^\n]{1,%d})" % max_len, t, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _find_city(t: str):
    # 1) explicit known cities
    for city in CITIES_HINT:
        if re.search(rf"\b{re.escape(city)}\b", t, re.IGNORECASE):
            return city
    # 2) pattern "DAP <City>" or "в г. <City>"
    m = re.search(r"\bDAP\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\- ]{2,30})\b", t)
    if m: return m.group(1).strip()
    m = re.search(r"\bг\.?\s*([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\- ]{2,30})\b", t)
    if m: return m.group(1).strip()
    # 3) pattern "<City>; Республика"
    m = re.search(r"\b([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\- ]{2,30})\s*;\s*Республика", t)
    if m: return m.group(1).strip()
    return None



def _extract_fields(text: str):
    """
    Strict extraction per user's rules, robust to spacing.
    ALWAYS returns the exact schema of 14 fields.
    """
    REQUIRED_COLS = ["Номер договора","Дата договора","Город","Продавец","Покупатель",
                     "ИИН/БИН клиента","IBAN","Номер карты","Сумма договора","Валюта",
                     "Срок с","Срок по","БИК банка","ИИК банка"]
    out = {k: None for k in REQUIRED_COLS}
    diag = {}

    t = text or ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"Д\s*О\s*Г\s*О\s*В\s*О\s*Р", "ДОГОВОР", t, flags=re.IGNORECASE)
    t = re.sub(r"\bN[°ºoо]?\s*", "№ ", t, flags=re.IGNORECASE)
    t = t.replace("–","-").replace("—","-").strip()

    m = re.search(r"ДОГОВОР\s*№\s*([A-Za-zА-Яа-я0-9/_\-]+)", t)
    if m:
        out["Номер договора"] = m.group(1); diag["Номер договора"] = "rx:ДОГОВОР №"

    m = re.search(r"ДОГОВОР\s*№\s*[A-Za-zА-Яа-я0-9/_\-]+\s*от\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})", t)
    if m:
        out["Дата договора"] = m.group(1); diag["Дата договора"] = "rx:№ ... от <дата>"

    city = None
    m = re.search(r"\bDAP\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\- ]{2,30})\b", t)
    if m:
        import re as _re
        city = _re.split(r"\s+(?=согласно\b)", m.group(1))[0].strip()
    if not city:
        m = re.search(r"\b([A-ZА-ЯЁ][A-Za-z-А-Яа-яЁё\- ]{2,30})\s*;\s*Республика\b", t)
        if m: city = m.group(1).strip()
    if not city:
        m = re.search(r"\bг\.?\s*([A-ZА-ЯЁ][A-Za-z-А-Яа-яЁё\- ]{2,30})\b", t)
        if m: city = m.group(1).strip()
    if city:
        out["Город"] = city; diag["Город"] = "pattern"

    m = re.search(r"Продавец\s*[:\-]\s*([^\n;:]{3,100})", t, re.IGNORECASE)
    if m:
        out["Продавец"] = m.group(1).strip(); diag["Продавец"] = "kw:Продавец:"

    m = re.search(r"(?:Покупатель|Клиент)\s*[:\-]\s*([^\n;:]{3,100})", t, re.IGNORECASE)
    if m:
        out["Покупатель"] = m.group(1).strip(); diag["Покупатель"] = "kw:Покупатель:/Клиент:"

    m = re.search(r"ИИН\s*[:№]?\s*([\d\s]{12,18})", t)
    if not m:
        m = re.search(r"БИН\s*[:№]?\s*([\d\s]{12,18})", t)
    if m:
        clean = re.sub(r"\D", "", m.group(1))
        if len(clean)==12:
            out["ИИН/БИН клиента"] = clean; diag["ИИН/БИН клиента"] = "rx:ИИН/БИН"

    m = re.search(r"\bKZ\s*\d{2}\s*[0-9A-Z]{3}\s*\d{13}\b", t, re.IGNORECASE)
    if m:
        out["IBAN"] = re.sub(r"\s+","", m.group(0)).upper(); diag["IBAN"] = "rx:KZ IBAN"

    for m in re.finditer(r"\b(?:\d[ \-]?){13,19}\b", t):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits)==16:
            out["Номер карты"] = f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"
            diag["Номер карты"] = "rx:16digits"; break

    m = re.search(r"(Сумма договора|Общая стоимость договора|Общая сумма договора|Стоимость договора)[^\d]{0,40}([\d\s.,]+)", t, re.IGNORECASE)
    if m:
        val = re.sub(r"[\s,]", "", m.group(2))
        try:
            out["Сумма договора"] = float(val); diag["Сумма договора"] = "rx:amount"
        except:
            pass

    m = re.search(r"Валюта договора\s*[:\-]?\s*([A-Za-zА-Яа-яЁё]+(?:\s+[A-Za-zА-Яа-яЁё]+)?)", t, re.IGNORECASE)
    if m:
        cur = m.group(1).strip().lower()
        if "тенге" in cur or "тнг" in cur: cur = "KZT"
        elif "руб" in cur: cur = "RUB"
        elif "доллар" in cur: cur = "USD"
        elif "евро" in cur: cur = "EUR"
        out["Валюта"] = cur.upper() if len(cur) <= 4 else cur
        diag["Валюта"] = "kw:Валюта договора"

    m_list = list(re.finditer(r"\bпо\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})", t, re.IGNORECASE))
    if m_list:
        out["Срок по"] = m_list[-1].group(1); diag["Срок по"] = "rx:по <дата>"

    if out["Дата договора"]:
        out["Срок с"] = out["Дата договора"]; diag["Срок с"] = "copy:Дата договора"

    return out, diag



def _to_xlsx_bytes(row: dict):
    cols = ["Номер договора","Дата договора","Город","Продавец","Покупатель","ИИН/БИН клиента","IBAN","Номер карты","Сумма договора","Валюта","Срок с","Срок по","БИК банка","ИИК банка"]
    row2 = {k: row.get(k) for k in cols}
    df = pd.DataFrame([row2], columns=cols)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="result")
    bio.seek(0)
    return bio.getvalue()

if uploaded is not None:
    raw = uploaded.read()
    if not raw:
        st.error("Файл пуст"); st.stop()
    if len(raw) > MAX_FILE_SIZE_MB*1024*1024:
        st.error("Файл слишком большой"); st.stop()

    pages = _decode_file_to_bgr_list(raw, uploaded.name, DEFAULTS["pdf_dpi"])
    if not pages:
        st.error("Не удалось прочитать документ"); st.stop()

    st.subheader("Все страницы: Оригинал vs Улучшено")
    for i, p in enumerate(pages, start=1):
        enh = _enhance(p, DEFAULTS["scale"], DEFAULTS["denoise_h"], DEFAULTS["sharp_amount"])
        c1, c2 = st.columns(2)
        with c1: _render_image(_shrink(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)), f"Стр. {i} — Оригинал")
        with c2: _render_image(_shrink(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)), f"Стр. {i} — Улучшено")

    st.subheader("Распознавание (EasyOCR)")
    with st.spinner("OCR..."):
        reader = easyocr.Reader(LANG_LIST)
        parts = []
        for p in pages:
            enh = _enhance(p, DEFAULTS["scale"], DEFAULTS["denoise_h"], DEFAULTS["sharp_amount"])
            res = reader.readtext(enh, detail=1, paragraph=False)
            for _, txt, conf in res:
                t = (txt or "").strip()
                if t and (conf or 0) >= DEFAULTS["ocr_threshold"]:
                    parts.append(t)
        full = "\n".join(parts)

    st.text_area("Текст (>= порога)", full, height=220)

    fields, diag = _extract_fields(full)
    st.subheader("Ключевые данные")
    st.json(fields)
    with st.expander("Диагностика соответствий", expanded=False):
        st.json(diag)

    xlsx_bytes = _to_xlsx_bytes(fields)
    st.download_button("⬇️ Скачать XLSX", xlsx_bytes, file_name="result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    json_bytes = json.dumps(fields, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ Скачать JSON", json_bytes, file_name="result.json", mime="application/json")
else:
    st.info("Загрузите документ. По умолчанию: масштаб 1.5×, шум 15, резкость 2.0, порог 0.85, DPI 300.")
