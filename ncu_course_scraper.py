"""
NCU Course Scraper — All Departments
======================================
Outputs 4 files:
  ncu_courses_en.jsonl      RAG chunks (one JSON per line, course names in Chinese)
  ncu_courses_zh.jsonl      RAG chunks in Chinese (one JSON per line)
  ncu_timeslot_lookup.json  Period number → time range  (e.g. "5" → "13:00–13:50")
  ncu_building_lookup.json  Building code → full name   (e.g. "E1" → "Engineering Building #2")

No API key required.

Usage:
  pip install requests beautifulsoup4
  python ncu_course_scraper.py
"""

import json, time, re, sys, os
import requests
from bs4 import BeautifulSoup

# ── No API key needed ──────────────────────────────────────────────────────────
# Translation is disabled. course_name_en / instructor_en will keep the original
# Chinese text. The RAG agent (ncu_rag_agent.py) handles understanding via
# Gemini embeddings which work well with mixed Chinese/English text.

TRANSLATE_BATCH_SIZE = 20


def translate_all(all_courses: list[dict]) -> list[dict]:
    """No translation — copy raw Chinese fields into the _en slots."""
    for c in all_courses:
        c["course_name_en"] = c.get("course_name", "")
        c["instructor_en"] = c.get("instructor", "")
        c["notes_en"] = c.get("notes", "")
        c["dept_name_en"] = c.get("dept_name_zh", "")
    return all_courses


# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL = "https://cis.ncu.edu.tw"
INDEX_URL = "https://cis.ncu.edu.tw/Course/main/query/byUnion"
TABLE_URL = "https://cis.ncu.edu.tw/Course/main/query/byUnion?dept={dept_id}&show=table"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": BASE_URL,
}

CRAWL_DELAY = 0.6

OUT_EN = "ncu_courses_en.jsonl"
OUT_ZH = "ncu_courses_zh.jsonl"
OUT_TIMESLOT = "ncu_timeslot_lookup.json"
OUT_BUILDING = "ncu_building_lookup.json"

# ── Official NCU time-slot table ───────────────────────────────────────────────
# Source: pdc.adm.ncu.edu.tw/Course/course/COUR_U.pdf
# Format in course data: weekday (一二三四五六日) + period codes (1-9, A-D, Z)
# e.g. "三5" = Wednesday period 5 = Wednesday 13:00–13:50

TIMESLOT_LOOKUP = {
    "1": {
        "time": "08:00–08:50",
        "time_zh": "08:00–08:50",
        "label_en": "Period 1",
        "label_zh": "第1節",
    },
    "2": {
        "time": "09:00–09:50",
        "time_zh": "09:00–09:50",
        "label_en": "Period 2",
        "label_zh": "第2節",
    },
    "3": {
        "time": "10:00–10:50",
        "time_zh": "10:00–10:50",
        "label_en": "Period 3",
        "label_zh": "第3節",
    },
    "4": {
        "time": "11:00–11:50",
        "time_zh": "11:00–11:50",
        "label_en": "Period 4",
        "label_zh": "第4節",
    },
    "Z": {
        "time": "12:00–12:50",
        "time_zh": "12:00–12:50",
        "label_en": "Period Z (noon)",
        "label_zh": "第Z節（午休）",
    },
    "5": {
        "time": "13:00–13:50",
        "time_zh": "13:00–13:50",
        "label_en": "Period 5",
        "label_zh": "第5節",
    },
    "6": {
        "time": "14:00–14:50",
        "time_zh": "14:00–14:50",
        "label_en": "Period 6",
        "label_zh": "第6節",
    },
    "7": {
        "time": "15:00–15:50",
        "time_zh": "15:00–15:50",
        "label_en": "Period 7",
        "label_zh": "第7節",
    },
    "8": {
        "time": "16:00–16:50",
        "time_zh": "16:00–16:50",
        "label_en": "Period 8",
        "label_zh": "第8節",
    },
    "9": {
        "time": "17:00–17:50",
        "time_zh": "17:00–17:50",
        "label_en": "Period 9",
        "label_zh": "第9節",
    },
    "A": {
        "time": "18:00–18:50",
        "time_zh": "18:00–18:50",
        "label_en": "Period 10",
        "label_zh": "第A節",
    },
    "B": {
        "time": "19:00–19:50",
        "time_zh": "19:00–19:50",
        "label_en": "Period 11",
        "label_zh": "第B節",
    },
    "C": {
        "time": "20:00–20:50",
        "time_zh": "20:00–20:50",
        "label_en": "Period 12",
        "label_zh": "第C節",
    },
    "D": {
        "time": "21:00–21:50",
        "time_zh": "21:00–21:50",
        "label_en": "Period 13",
        "label_zh": "第D節",
    },
}

WEEKDAY_MAP = {
    "一": {"en": "Monday", "zh": "星期一"},
    "二": {"en": "Tuesday", "zh": "星期二"},
    "三": {"en": "Wednesday", "zh": "星期三"},
    "四": {"en": "Thursday", "zh": "星期四"},
    "五": {"en": "Friday", "zh": "星期五"},
    "六": {"en": "Saturday", "zh": "星期六"},
    "日": {"en": "Sunday", "zh": "星期日"},
}

# ── Official NCU building codes ────────────────────────────────────────────────
# Source: COUR_U.pdf + CCOP_EL.pdf + university.reviewiki.com (confirmed)
# Format: building prefix → { en, zh } names

BUILDING_LOOKUP = {
    # Engineering cluster
    "E": {"en": "Engineering Building #1 (工程一館)", "zh": "工程一館"},
    "E1": {
        "en": "Engineering Building #2 / EECS Office (工程二館)",
        "zh": "工程二館（資電學院辦公室）",
    },
    "E2": {"en": "Engineering Building #3 (工程三館)", "zh": "工程三館"},
    "E3": {"en": "Engineering Building #4 (工程四館)", "zh": "工程四館"},
    "E4": {"en": "Engineering Building #5 (工程五館)", "zh": "工程五館"},
    "E5": {"en": "Engineering Building #6 (工程六館)", "zh": "工程六館"},
    "E6": {"en": "Engineering Building #7 (工程七館)", "zh": "工程七館"},
    # Science cluster
    "S": {"en": "Science Building (科學館)", "zh": "科學館"},
    "S2": {"en": "Science Building #3 (科學三館)", "zh": "科學三館"},
    "S3": {"en": "Science Building #4 (科學四館)", "zh": "科學四館"},
    # Management & Humanities
    "M": {"en": "Management Building (管理學院大樓)", "zh": "管理學院大樓"},
    "H": {"en": "Humanities Building (人文館)", "zh": "人文館"},
    "HA": {"en": "Humanities Building A (人文館A棟)", "zh": "人文館A棟"},
    "HB": {"en": "Humanities Building B (人文館B棟)", "zh": "人文館B棟"},
    # General classrooms (numeric prefix codes from reviewiki)
    "10": {"en": "Baisha Building (白沙大樓)", "zh": "白沙大樓"},
    "11": {"en": "Mingde Hall (明德館)", "zh": "明德館"},
    "12": {"en": "Zhishan Hall (至善館)", "zh": "至善館"},
    "13": {"en": "Jieying Hall (擷英館)", "zh": "擷英館"},
    "14": {"en": "Hongdao Hall (弘道館)", "zh": "弘道館"},
    "15": {"en": "Yihui Hall (藝薈館)", "zh": "藝薈館"},
    "16": {"en": "Shengyang Hall (聲洋館)", "zh": "聲洋館"},
    # Other
    "ST": {"en": "Student Activity Center (學生活動中心)", "zh": "學生活動中心"},
    "LI": {"en": "Library (圖書館)", "zh": "圖書館"},
    "GY": {"en": "Gymnasium (體育館)", "zh": "體育館"},
    "AT": {"en": "Astronomical Observatory (天文台)", "zh": "天文台"},
    "R": {"en": "Ren-Sheng Building (仁生館)", "zh": "仁生館"},
    "C": {"en": "Computer Center (電算中心)", "zh": "電算中心"},
    "EL": {"en": "Language Center (語言中心)", "zh": "語言中心"},
}

# ── Column mapping (zh → en key) ──────────────────────────────────────────────

COLUMN_MAP = {
    "流水號": "serial_number",
    "課號": "course_code",
    "班次": "class_section",
    "課程名稱": "course_name",
    "教師": "instructor",
    "學分": "credits",
    "必/選修": "required_elective",
    "上課時間": "schedule_raw",
    "上課地點": "classroom_raw",
    "選課人數": "enrollment",
    "備註": "notes",
}

REQUIRED_ELECTIVE_MAP = {
    "必": {"en": "Required", "zh": "必修"},
    "選": {"en": "Elective", "zh": "選修"},
    "必選": {"en": "Required/Elective", "zh": "必選修"},
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch(url: str, session: requests.Session) -> BeautifulSoup:
    time.sleep(CRAWL_DELAY)
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def resolve_building(code: str) -> dict:
    """
    Given a classroom code like 'E6-301', 'S311', 'E1204', return building info.
    Tries longer prefixes first (E6 before E).
    """
    if not code or code in ("***", ""):
        return {}
    # strip room number: take leading letters/digits that form the building prefix
    # e.g. "E6-301" → try "E6", then "E"
    #      "S2-107" → try "S2", then "S"
    #      "10101"  → try "10"
    prefix = re.match(r"^([A-Za-z]+\d*|[0-9]{2})", code)
    if not prefix:
        return {}
    raw = prefix.group(1)
    # try progressively shorter prefixes
    for length in range(len(raw), 0, -1):
        key = raw[:length].upper()
        if key in BUILDING_LOOKUP:
            return {"building_code": key, **BUILDING_LOOKUP[key]}
    return {"building_code": raw}


def parse_schedule(raw: str) -> list[dict]:
    """
    Parse a schedule string like '三56' or '一34 四78' into structured slots.
    Returns list of { weekday_zh, weekday_en, periods, times_en, times_zh }
    """
    if not raw or raw in ("***", ""):
        return []
    slots = []
    # pattern: one CJK weekday char followed by period codes
    for m in re.finditer(r"([一二三四五六日])([1-9A-DZ]+)", raw):
        day_char = m.group(1)
        period_str = m.group(2)
        day_info = WEEKDAY_MAP.get(day_char, {"en": day_char, "zh": day_char})
        periods = []
        times_en = []
        times_zh = []
        for p in period_str:
            p_upper = p.upper()
            info = TIMESLOT_LOOKUP.get(p_upper, {})
            periods.append(p_upper)
            times_en.append(info.get("time", p_upper))
            times_zh.append(info.get("time_zh", p_upper))
        slots.append(
            {
                "weekday_zh": day_info["zh"],
                "weekday_en": day_info["en"],
                "periods": periods,
                "times_en": times_en,
                "times_zh": times_zh,
            }
        )
    return slots


def schedule_to_text_en(slots: list[dict]) -> str:
    if not slots:
        return ""
    parts = []
    for s in slots:
        time_range = ", ".join(s["times_en"])
        parts.append(f"{s['weekday_en']} ({time_range})")
    return "; ".join(parts)


def schedule_to_text_zh(slots: list[dict]) -> str:
    if not slots:
        return ""
    parts = []
    for s in slots:
        time_range = "、".join(s["times_zh"])
        parts.append(f"{s['weekday_zh']}（{time_range}）")
    return "；".join(parts)


# ── Department discovery ───────────────────────────────────────────────────────


def get_departments(session: requests.Session) -> list[dict]:
    print(f"Fetching department list from {INDEX_URL} ...")
    soup = fetch(INDEX_URL, session)
    depts = []

    for select in soup.find_all("select"):
        for opt in select.find_all("option"):
            val = opt.get("value", "").strip()
            label = clean(opt.get_text())
            if val and val.startswith("dept") and label:
                depts.append({"id": val, "name_zh": label, "name_en": label})

    if depts:
        print(f"  Found {len(depts)} departments from <select>.")
        return depts

    seen = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"dept=(dept[^&\"'\s]+)", a["href"])
        if m:
            dept_id = m.group(1)
            if dept_id not in seen:
                seen.add(dept_id)
                label = clean(a.get_text()) or dept_id
                depts.append({"id": dept_id, "name_zh": label, "name_en": label})

    if depts:
        print(f"  Found {len(depts)} departments from links.")
        return depts

    print(
        "  [WARN] Could not auto-discover departments — falling back to single known dept."
    )
    return [
        {
            "id": "deptI1I5002I0",
            "name_zh": "資訊工程學系",
            "name_en": "Dept. of Computer Science & Engineering (fallback)",
        }
    ]


# ── Table parser ───────────────────────────────────────────────────────────────


def parse_table(soup: BeautifulSoup, dept: dict) -> list[dict]:
    table = None
    for t in soup.find_all("table"):
        if t.find("th") or (t.find("tr") and len(t.find_all("tr")) > 1):
            table = t
            break
    if not table:
        return []

    header_row = table.find("tr")
    if not header_row:
        return []

    raw_headers = [clean(th.get_text()) for th in header_row.find_all(["th", "td"])]
    keys = [COLUMN_MAP.get(h, h) for h in raw_headers]

    courses = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        record = {
            "dept_id": dept["id"],
            "dept_name_zh": dept["name_zh"],
            "dept_name_en": "",  # filled in by translate_all()
        }

        for i, key in enumerate(keys):
            if i >= len(cells):
                record[key] = ""
                continue
            cell = cells[i]
            text = clean(cell.get_text())
            link = cell.find("a")
            record[key] = text
            if link and link.get("href"):
                href = link["href"]
                if href.startswith("/"):
                    href = BASE_URL + href
                record[key + "_url"] = href

        non_empty = [
            v
            for k, v in record.items()
            if isinstance(v, str)
            and v
            and k not in ("dept_id", "dept_name_zh", "dept_name_en")
        ]
        if not non_empty:
            continue

        # ── Enrich: schedule ──────────────────────────────────────────────────
        raw_sched = record.get("schedule_raw", "")
        slots = parse_schedule(raw_sched)
        record["schedule_parsed"] = slots
        record["schedule_text_en"] = schedule_to_text_en(slots)
        record["schedule_text_zh"] = schedule_to_text_zh(slots)

        # ── Enrich: extract dept prefix from course code ──────────────────────
        # course_code like "CS3001" → dept_prefix "CS"
        # used by the agent's _augment_query to expand dept abbreviations
        course_code = record.get("course_code", "")
        prefix_match = re.match(r"^([A-Za-z]+)", course_code)
        record["dept_code_prefix"] = (
            prefix_match.group(1).upper() if prefix_match else ""
        )

        # ── Enrich: classroom → building ──────────────────────────────────────
        raw_room = record.get("classroom_raw", "")
        building = resolve_building(raw_room)
        record["building_info"] = building

        # ── Enrich: required/elective ─────────────────────────────────────────
        req_raw = record.get("required_elective", "")
        req_info = REQUIRED_ELECTIVE_MAP.get(req_raw, {"en": req_raw, "zh": req_raw})
        record["required_elective_en"] = req_info["en"]
        record["required_elective_zh"] = req_info["zh"]

        courses.append(record)

    return courses


# ── RAG text formatters ────────────────────────────────────────────────────────


def to_text_en(c: dict) -> str:
    lines = []
    name = c.get("course_name_en") or c.get("course_name", "")
    serial = c.get("serial_number", "")
    code = c.get("course_code", "")
    section = c.get("class_section", "")
    prefix = c.get("dept_code_prefix", "")
    dept = c.get("dept_name_en", "")
    inst = c.get("instructor_en") or c.get("instructor", "")
    cred = c.get("credits", "")
    req = c.get("required_elective_en", "")
    sched = c.get("schedule_text_en", "") or c.get("schedule_raw", "")
    room = c.get("classroom_raw", "")
    bld = c.get("building_info", {})
    enrol = c.get("enrollment", "")
    notes = c.get("notes_en") or c.get("notes", "")

    header = f"Course: {name}"
    if code:
        header += f" [{code}"
        if section:
            header += f"-{section}"
        header += "]"
    lines.append(header)
    if serial:
        lines.append(f"Serial Number: {serial}")
    if prefix:
        lines.append(f"Dept. Code Prefix: {prefix} (from course code {code})")
    if dept:
        lines.append(f"Department: {dept}")
    if inst:
        lines.append(f"Instructor: {inst}")
    if cred:
        lines.append(f"Credits: {cred}")
    if req:
        lines.append(f"Type: {req}")
    if sched:
        lines.append(f"Schedule: {sched}")
    if room:
        room_line = f"Classroom: {room}"
        if bld.get("en"):
            room_line += f" — {bld['en']}"
        lines.append(room_line)
    if enrol:
        lines.append(f"Enrollment: {enrol}")
    if notes:
        lines.append(f"Notes: {notes}")
    url = c.get("course_name_url", "")
    if url:
        lines.append(f"Detail URL: {url}")
    return "\n".join(lines)


def to_text_zh(c: dict) -> str:
    lines = []
    name = c.get("course_name", "")
    serial = c.get("serial_number", "")
    code = c.get("course_code", "")
    section = c.get("class_section", "")
    dept = c.get("dept_name_zh", "")
    inst = c.get("instructor", "")
    cred = c.get("credits", "")
    req = c.get("required_elective_zh", "")
    sched = c.get("schedule_text_zh", "") or c.get("schedule_raw", "")
    room = c.get("classroom_raw", "")
    bld = c.get("building_info", {})
    enrol = c.get("enrollment", "")
    notes = c.get("notes", "")

    header = f"課程：{name}"
    if code:
        header += f"（{code}"
        if section:
            header += f"-{section}"
        header += "）"
    lines.append(header)
    if serial:
        lines.append(f"流水號：{serial}")
    if dept:
        lines.append(f"開課系所：{dept}")
    if inst:
        lines.append(f"授課教師：{inst}")
    if cred:
        lines.append(f"學分數：{cred}")
    if req:
        lines.append(f"修課性質：{req}")
    if sched:
        lines.append(f"上課時間：{sched}")
    if room:
        room_line = f"上課地點：{room}"
        if bld.get("zh"):
            room_line += f"（{bld['zh']}）"
        lines.append(room_line)
    if enrol:
        lines.append(f"選課人數：{enrol}")
    if notes:
        lines.append(f"備註：{notes}")
    url = c.get("course_name_url", "")
    if url:
        lines.append(f"課程詳細頁面：{url}")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # Save lookup tables immediately so they're available even if scraping fails
    with open(OUT_TIMESLOT, "w", encoding="utf-8") as f:
        json.dump(TIMESLOT_LOOKUP, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {OUT_TIMESLOT}")

    with open(OUT_BUILDING, "w", encoding="utf-8") as f:
        json.dump(BUILDING_LOOKUP, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {OUT_BUILDING}\n")

    session = requests.Session()
    departments = get_departments(session)
    print(f"Will scrape {len(departments)} department(s).\n")

    all_courses = []
    failed_depts = []

    for i, dept in enumerate(departments):
        url = TABLE_URL.format(dept_id=dept["id"])
        print(f"[{i+1}/{len(departments)}] {dept['name_zh']}  ({dept['id']})")
        try:
            soup = fetch(url, session)
            courses = parse_table(soup, dept)
            print(f"  -> {len(courses)} courses")
            all_courses.extend(courses)
        except requests.HTTPError as e:
            print(f"  [ERROR] HTTP {e.response.status_code} — skipping")
            failed_depts.append(dept)
        except Exception as e:
            print(f"  [ERROR] {e} — skipping")
            failed_depts.append(dept)

    print(f"\nTotal courses scraped: {len(all_courses)}")

    if not all_courses:
        print("\nNo courses found. Possible causes:")
        print("  1. Site blocking requests — add browser Cookie to HEADERS dict")
        print("  2. JavaScript-rendered page — install playwright and rewrite fetch()")
        print("  3. HTML structure changed — inspect page and adjust parse_table()")
        sys.exit(1)

    # Translate all Chinese text fields to English
    all_courses = translate_all(all_courses)

    # English JSONL
    with open(OUT_EN, "w", encoding="utf-8") as f:
        for c in all_courses:
            c["_text_en"] = to_text_en(c)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved -> {OUT_EN}  ({len(all_courses)} records)")

    # Chinese JSONL
    with open(OUT_ZH, "w", encoding="utf-8") as f:
        for c in all_courses:
            c["_text_zh"] = to_text_zh(c)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved -> {OUT_ZH}  ({len(all_courses)} records)")

    if failed_depts:
        print(f"\n[WARN] {len(failed_depts)} department(s) failed:")
        for d in failed_depts:
            print(f"  {d['name_zh']}  ({d['id']})")

    print("\nAll done. Output files:")
    print(f"  {OUT_EN:<30} English RAG chunks (vector DB ready)")
    print(f"  {OUT_ZH:<30} Chinese RAG chunks (vector DB ready)")
    print(f"  {OUT_TIMESLOT:<30} Period number → time range lookup")
    print(f"  {OUT_BUILDING:<30} Building code → full name lookup")


if __name__ == "__main__":
    main()
