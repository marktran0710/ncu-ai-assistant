"""
NCU Course Scraper — https://cis.ncu.edu.tw/Course/main/query/byUnion
=======================================================================
Output → ./data/
  courses_en.jsonl / courses_zh.jsonl / courses_combined.jsonl / courses_raw.jsonl
  timeslot_lookup.json / building_lookup.json / scrape_report.json

Usage:
  pip install requests beautifulsoup4 lxml
  python ncu_course_scraper.py
  python ncu_course_scraper.py --dept deptI1I5002I0   # debug 1 dept
  python ncu_course_scraper.py --delay 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR      = Path("data")
OUT_EN        = DATA_DIR / "courses_en.jsonl"
OUT_ZH        = DATA_DIR / "courses_zh.jsonl"
OUT_COMBINED  = DATA_DIR / "courses_combined.jsonl"   # bilingual prose — best for embedding
OUT_RAW       = DATA_DIR / "courses_raw.jsonl"
OUT_TIMESLOT  = DATA_DIR / "timeslot_lookup.json"
OUT_BUILDING  = DATA_DIR / "building_lookup.json"
OUT_REPORT    = DATA_DIR / "scrape_report.json"

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
BASE_URL  = "https://cis.ncu.edu.tw"
INDEX_URL = f"{BASE_URL}/Course/main/query/byUnion"
TABLE_URL = f"{BASE_URL}/Course/main/query/byUnion?dept={{dept_id}}&show=table"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
    "Referer":         INDEX_URL,
}

DEFAULT_DELAY = 0.8
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
TIMESLOT_LOOKUP: dict[str, dict] = {
    "1": {"time": "08:00-08:50", "label_en": "Period 1",        "label_zh": "第1節"},
    "2": {"time": "09:00-09:50", "label_en": "Period 2",        "label_zh": "第2節"},
    "3": {"time": "10:00-10:50", "label_en": "Period 3",        "label_zh": "第3節"},
    "4": {"time": "11:00-11:50", "label_en": "Period 4",        "label_zh": "第4節"},
    "Z": {"time": "12:00-12:50", "label_en": "Period Z (noon)", "label_zh": "第Z節（午休）"},
    "5": {"time": "13:00-13:50", "label_en": "Period 5",        "label_zh": "第5節"},
    "6": {"time": "14:00-14:50", "label_en": "Period 6",        "label_zh": "第6節"},
    "7": {"time": "15:00-15:50", "label_en": "Period 7",        "label_zh": "第7節"},
    "8": {"time": "16:00-16:50", "label_en": "Period 8",        "label_zh": "第8節"},
    "9": {"time": "17:00-17:50", "label_en": "Period 9",        "label_zh": "第9節"},
    "A": {"time": "18:00-18:50", "label_en": "Period A",        "label_zh": "第A節"},
    "B": {"time": "19:00-19:50", "label_en": "Period B",        "label_zh": "第B節"},
    "C": {"time": "20:00-20:50", "label_en": "Period C",        "label_zh": "第C節"},
    "D": {"time": "21:00-21:50", "label_en": "Period D",        "label_zh": "第D節"},
}

WEEKDAY_ZH_TO_EN: dict[str, str] = {
    "日": "Sunday",  "一": "Monday",   "二": "Tuesday",
    "三": "Wednesday","四": "Thursday", "五": "Friday",  "六": "Saturday",
}

BUILDING_LOOKUP: dict[str, dict] = {
    "E":   {"en": "Engineering Building 1",           "zh": "工程一館"},
    "E1":  {"en": "Engineering Building 2 / EECS",    "zh": "工程二館"},
    "E2":  {"en": "Engineering Building 3",           "zh": "工程三館"},
    "E3":  {"en": "Engineering Building 4",           "zh": "工程四館"},
    "E4":  {"en": "Engineering Building 5",           "zh": "工程五館"},
    "E5":  {"en": "Engineering Building 6",           "zh": "工程六館"},
    "E6":  {"en": "Engineering Building 7",           "zh": "工程七館"},
    "S":   {"en": "Science Building",                 "zh": "科學館"},
    "S2":  {"en": "Science Building 3",               "zh": "科學三館"},
    "S3":  {"en": "Science Building 4",               "zh": "科學四館"},
    "M":   {"en": "Management Building",              "zh": "管理學院大樓"},
    "H":   {"en": "Humanities Building",              "zh": "人文館"},
    "HA":  {"en": "Humanities Building A",            "zh": "人文館A棟"},
    "HB":  {"en": "Humanities Building B",            "zh": "人文館B棟"},
    "10":  {"en": "Baisha Building",                  "zh": "白沙大樓"},
    "11":  {"en": "Mingde Hall",                      "zh": "明德館"},
    "12":  {"en": "Zhishan Hall",                     "zh": "至善館"},
    "13":  {"en": "Jieying Hall",                     "zh": "擷英館"},
    "14":  {"en": "Hongdao Hall",                     "zh": "弘道館"},
    "15":  {"en": "Yihui Hall",                       "zh": "藝薈館"},
    "16":  {"en": "Shengyang Hall",                   "zh": "聲洋館"},
    "ST":  {"en": "Student Activity Center",          "zh": "學生活動中心"},
    "LI":  {"en": "Library",                          "zh": "圖書館"},
    "GY":  {"en": "Gymnasium",                        "zh": "體育館"},
    "AT":  {"en": "Astronomical Observatory",         "zh": "天文台"},
    "R":   {"en": "Ren-Sheng Building",               "zh": "仁生館"},
    "C":   {"en": "Computer Center",                  "zh": "電算中心"},
    "EL":  {"en": "Language Center",                  "zh": "語言中心"},
    "SP":  {"en": "Sports Center",                    "zh": "體育中心"},
    "LS":  {"en": "Life Science Building",            "zh": "生命科學館"},
    "ES":  {"en": "Earth Science Building",           "zh": "地球科學館"},
    "OP":  {"en": "Optics Building",                  "zh": "光電館"},
    "RC":  {"en": "Ren-Sheng Building (RC wing)",     "zh": "仁生館RC"},
}

REQUIRED_ELECTIVE_MAP: dict[str, dict] = {
    "必修": {"en": "Required",          "zh": "必修"},
    "選修": {"en": "Elective",          "zh": "選修"},
    "必選": {"en": "Required/Elective", "zh": "必選修"},
    "通識": {"en": "General Education", "zh": "通識"},
    "必":   {"en": "Required",          "zh": "必修"},
    "選":   {"en": "Elective",          "zh": "選修"},
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScheduleSlot:
    weekday_zh: str
    weekday_en: str
    periods:    list[str]
    times:      list[str]
    classroom:  str = ""
    building_en:str = ""
    building_zh:str = ""

    def text_en(self) -> str:
        times_str = ", ".join(self.times)
        room = f" in room {self.classroom}" if self.classroom else ""
        bld  = f" ({self.building_en})"     if self.building_en else ""
        return f"{self.weekday_en} periods {','.join(self.periods)} ({times_str}){room}{bld}"

    def text_zh(self) -> str:
        times_str = "、".join(self.times)
        room = f" {self.classroom}"        if self.classroom else ""
        bld  = f"（{self.building_zh}）"   if self.building_zh else ""
        return f"{self.weekday_zh} 第{''.join(self.periods)}節（{times_str}）{room}{bld}"


@dataclass
class Course:
    dept_id:              str = ""
    dept_name_zh:         str = ""
    serial_number:        str = ""
    course_code:          str = ""
    class_section:        str = ""
    course_name_zh:       str = ""
    course_name_en:       str = ""
    course_url:           str = ""
    instructor:           str = ""
    credits:              str = ""
    required_elective:    str = ""
    required_elective_en: str = ""
    required_elective_zh: str = ""
    full_half:            str = ""
    enrollment_limit:     str = ""
    notes:                str = ""
    schedule_slots:       list[ScheduleSlot] = field(default_factory=list)
    dept_code_prefix:     str = ""

    # ── Shared metadata dict used by all index doc builders ──────────────────
    def _metadata(self) -> dict:
        """Structured fields used for filtering — independent of text format."""
        return {
            "course_code":    self.course_code,
            "course_name":    self.course_name_en or self.course_name_zh,
            "course_name_zh": self.course_name_zh,
            "course_name_en": self.course_name_en,
            "instructor":     self.instructor,
            "dept_id":        self.dept_id,
            "dept_name_zh":   self.dept_name_zh,
            "dept_prefix":    self.dept_code_prefix,
            "credits":        self.credits,
            "enrollment":     self.enrollment_limit,
            "classrooms":     list({s.classroom    for s in self.schedule_slots if s.classroom}),
            "buildings":      list({s.building_en  for s in self.schedule_slots if s.building_en}),
            "weekdays":       list({s.weekday_en.lower() for s in self.schedule_slots}),
            "periods":        list({p for s in self.schedule_slots for p in s.periods}),
            "course_url":     self.course_url,
        }

    # ── Prose text builders ───────────────────────────────────────────────────

    def to_rag_text_en(self) -> str:
        """
        Natural prose in English — better for embedding than label-colon format.

        Example output:
          Engineering Mathematics (工程數學) is a 3-credit required course [I2010]
          offered by the Department of Computer Science and Engineering (資訊工程學系).
          It is taught by Wang Ming and meets on Monday and Wednesday,
          periods 3 and 4 (10:00–11:50) in room E6-A207 (Engineering Building 7).
        """
        name_en = self.course_name_en or self.course_name_zh
        name_zh = self.course_name_zh

        # ── Opening sentence ─────────────────────────────────────────────────
        bilingual_name = (
            f"{name_en} ({name_zh})" if name_en and name_zh and name_en != name_zh
            else name_en
        )
        code_part = f" [{self.course_code}]" if self.course_code else ""
        credits_part = (
            f"{self.credits}-credit " if self.credits else ""
        )
        req_part = (
            self.required_elective_en.lower() + " " if self.required_elective_en else ""
        )
        opening = f"{bilingual_name}{code_part} is a {credits_part}{req_part}course"

        # ── Department ───────────────────────────────────────────────────────
        if self.dept_name_zh:
            opening += f" offered by {self.dept_name_zh}"

        opening += "."

        # ── Instructor sentence ───────────────────────────────────────────────
        instructor_sent = ""
        if self.instructor:
            instructor_sent = f" It is taught by {self.instructor}."

        # ── Schedule sentence ─────────────────────────────────────────────────
        schedule_sent = ""
        if self.schedule_slots:
            slot_phrases = []
            for s in self.schedule_slots:
                periods_str  = " and ".join(s.periods) if len(s.periods) > 1 else (s.periods[0] if s.periods else "")
                time_range   = f"{s.times[0].split('-')[0]}–{s.times[-1].split('-')[1]}" if s.times else ""
                room_phrase  = f" in room {s.classroom}" if s.classroom else ""
                bld_phrase   = f" ({s.building_en})"     if s.building_en else ""
                slot_phrases.append(
                    f"{s.weekday_en} period{'s' if len(s.periods) != 1 else ''} "
                    f"{periods_str} ({time_range}){room_phrase}{bld_phrase}"
                )
            schedule_sent = " The course meets " + "; and ".join(slot_phrases) + "."

        # ── Enrollment / notes ────────────────────────────────────────────────
        extra_parts = []
        if self.enrollment_limit:
            extra_parts.append(f"enrollment limit {self.enrollment_limit}")
        if self.full_half:
            extra_parts.append(f"{self.full_half} semester")
        if self.notes:
            extra_parts.append(self.notes)
        extra_sent = (" " + "; ".join(extra_parts).capitalize() + ".") if extra_parts else ""

        return opening + instructor_sent + schedule_sent + extra_sent

    def to_rag_text_zh(self) -> str:
        """
        Natural prose in Chinese — mirrors to_rag_text_en() structure.

        Example output:
          工程數學（Engineering Mathematics）[I2010] 是資訊工程學系開設的
          3學分必修課，由王明老師授課。本課程於每週一及週三第3、4節
          （10:00–11:50）在E6-A207教室（工程七館）上課。
        """
        name_zh = self.course_name_zh
        name_en = self.course_name_en
        bilingual_name = (
            f"{name_zh}（{name_en}）" if name_zh and name_en and name_zh != name_en
            else name_zh or name_en
        )
        code_part   = f"[{self.course_code}]" if self.course_code else ""
        credits_str = f"{self.credits}學分" if self.credits else ""
        req_str     = self.required_elective_zh if self.required_elective_zh else ""

        # Opening
        type_str = "、".join(filter(None, [credits_str, req_str]))
        opening = f"{bilingual_name}{code_part}"
        if self.dept_name_zh:
            opening += f" 是{self.dept_name_zh}開設的{type_str}課程。"
        else:
            opening += f" 是{type_str}課程。"

        # Instructor
        instructor_sent = f"本課程由{self.instructor}老師授課。" if self.instructor else ""

        # Schedule
        schedule_sent = ""
        if self.schedule_slots:
            slot_phrases = []
            for s in self.schedule_slots:
                periods_str = "、".join(s.periods)
                time_range  = f"{s.times[0].split('-')[0]}–{s.times[-1].split('-')[1]}" if s.times else ""
                room_phrase = f"在{s.classroom}教室"    if s.classroom else ""
                bld_phrase  = f"（{s.building_zh}）"    if s.building_zh else ""
                slot_phrases.append(
                    f"每週{s.weekday_zh}第{periods_str}節（{time_range}）{room_phrase}{bld_phrase}"
                )
                schedule_sent = "上課時間為" + "；".join(slot_phrases) + "。"

        # Extra
        extra_parts = []
        if self.enrollment_limit:
            extra_parts.append(f"人數限制{self.enrollment_limit}人")
        if self.full_half:
            extra_parts.append(f"{self.full_half}年課")
        if self.notes:
            extra_parts.append(self.notes)
        extra_sent = ("備註：" + "；".join(extra_parts) + "。") if extra_parts else ""

        return opening + instructor_sent + schedule_sent + extra_sent

    def to_rag_text_combined(self) -> str:
        """
        Bilingual prose document — recommended for embedding.

        Merges English and Chinese prose with a separator so one vector
        captures both language representations. Cross-lingual queries
        (Chinese question → English-named course) resolve more reliably.
        """
        return self.to_rag_text_en() + "\n\n" + self.to_rag_text_zh()

    # ── Index doc builders ────────────────────────────────────────────────────

    def to_index_doc_en(self) -> dict:
        return {"text": self.to_rag_text_en(), **self._metadata()}

    def to_index_doc_zh(self) -> dict:
        # Override a few metadata fields with ZH-specific values
        meta = self._metadata()
        meta["buildings"] = list({s.building_zh for s in self.schedule_slots if s.building_zh})
        meta["weekdays"]  = list({s.weekday_zh  for s in self.schedule_slots})
        meta["course_name"] = self.course_name_zh
        return {"text": self.to_rag_text_zh(), **meta}

    def to_index_doc_combined(self) -> dict:
        """Best document for embedding — bilingual prose + full EN metadata."""
        return {"text": self.to_rag_text_combined(), **self._metadata()}

    def to_raw_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def resolve_building(classroom: str) -> tuple[str, str]:
    if not classroom:
        return "", ""
    paren    = re.search(r"[（(]([^）)]+)[）)]", classroom)
    embed_zh = paren.group(1).strip() if paren else ""
    code     = re.split(r"[\s（(]", classroom)[0].strip()
    prefix_m = re.match(r"^([A-Za-z]{1,2}\d?|[0-9]{2})", code)
    if not prefix_m:
        return "", embed_zh
    raw = prefix_m.group(1).upper()
    for length in range(len(raw), 0, -1):
        key = raw[:length]
        if key in BUILDING_LOOKUP:
            info = BUILDING_LOOKUP[key]
            return info["en"], embed_zh or info["zh"]
    return "", embed_zh


def parse_periods_and_classroom(cell_text: str) -> tuple[list[str], str]:
    lines     = [l.strip() for l in cell_text.strip().splitlines() if l.strip()]
    periods:  list[str] = []
    classroom = ""
    for line in lines:
        if re.fullmatch(r"[1-9A-DZa-dz]+", line):
            periods = list(line.upper())
        elif re.match(r"^[A-Za-z0-9]", line) and not re.fullmatch(r"\d+", line):
            classroom = line
        elif re.fullmatch(r"\d+", line) and all(d in "123456789" for d in line):
            periods = list(line)
    return periods, classroom


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def fetch(
    url: str,
    session: requests.Session,
    delay: float = DEFAULT_DELAY,
    retries: int = MAX_RETRIES,
) -> BeautifulSoup:
    time.sleep(delay)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=25)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return BeautifulSoup(resp.text, "lxml")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        wait = RETRY_BACKOFF * (2 ** (attempt - 1))
        logger.warning(f"  Attempt {attempt}/{retries} failed. Retry in {wait:.1f}s")
        time.sleep(wait)
    raise RuntimeError(f"All retries failed for {url}") from last_exc


# ---------------------------------------------------------------------------
# Department discovery
# ---------------------------------------------------------------------------
CEECS_KEYWORDS = [
    "電機", "資訊工程", "通訊工程", "網路學習", "照明顯示",
    "Electrical", "Computer Science", "Communication Engineering", "Network Learning",
]
CEECS_ID_PREFIXES = [
    "dept350",  # Electrical Engineering
    "dept520",  # Computer Science (CSIE)
    "dept355",  # Communication Engineering
    "dept357",  # Network Learning Technology
    "dept358",  # AI Graduate Program
]


def get_departments(session: requests.Session, delay: float = DEFAULT_DELAY) -> list[dict]:
    logger.info("Fetching department list and filtering for CEECS…")
    soup  = fetch(INDEX_URL, session, delay=delay)
    seen:  set[str]   = set()
    depts: list[dict] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m    = re.search(r"[?&]dept=(dept[^&\"'\s]+)", href)
        if not m:
            continue
        dept_id = m.group(1)
        if dept_id in seen:
            continue
        label = _clean(a.get_text())
        label = re.sub(r"\s*\(\d+\)\s*$", "", label).strip()

        is_ceecs_id   = any(dept_id.startswith(p) for p in CEECS_ID_PREFIXES)
        is_ceecs_name = any(kw in label for kw in CEECS_KEYWORDS)
        if is_ceecs_id or is_ceecs_name:
            seen.add(dept_id)
            depts.append({"id": dept_id, "name_zh": label})
            logger.debug(f"  Matched CEECS dept: {label} ({dept_id})")

    logger.info(f"  Discovered {len(depts)} CEECS departments.")
    return depts


# ---------------------------------------------------------------------------
# Table parser
# ---------------------------------------------------------------------------
WEEKDAY_COLS = {"日", "一", "二", "三", "四", "五", "六"}

HEADER_KEY_MAP: dict[str, str] = {
    "流水號":        "serial",
    "課號":          "code",
    "班次":          "section",
    "班別":          "section",
    "課程名稱":      "name",
    "課程名稱/備註": "name",
    "授課教師":      "instructor",
    "教師":          "instructor",
    "選修別":        "req",
    "必/選修":       "req",
    "修別":          "req",
    "學分":          "credits",
    "全/半":         "fullhalf",
    "人數限制":      "limit",
    "選課人數":      "limit",
    "時間/教室":     "timecls",
    "分發條件":      "dist",
}


def _find_main_table(soup: BeautifulSoup) -> Tag | None:
    best: Tag | None = None
    best_score = 0
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header_cells = rows[0].find_all(["th", "td"])
        headers      = [_clean(c.get_text()) for c in header_cells]
        score        = sum(1 for h in headers if h in HEADER_KEY_MAP or h in WEEKDAY_COLS)
        if score > best_score:
            best_score = score
            best       = table
    return best if best_score >= 3 else None


def _parse_name_cell(cell: Tag) -> tuple[str, str, str, str]:
    link = cell.find("a", href=True)
    course_url = ""
    if link:
        href       = link["href"]
        course_url = (BASE_URL + href) if href.startswith("/") else href

    raw_lines = [_clean(t) for t in cell.stripped_strings]
    noise     = {"密碼卡", "預選", "採用密碼卡", "部份使用", "全部使用", "[預選]"}

    name_zh, name_en, notes_parts = "", "", []
    for line in raw_lines:
        if not line:
            continue
        clean_line = re.sub(r"[\[【\(（][^\]】\)）]*[\]】\)）]", "", line).strip()
        if not clean_line:
            continue
        ascii_ratio = sum(1 for c in clean_line if c.isascii() and c.isalpha()) / max(len(clean_line), 1)
        if not name_zh and ascii_ratio < 0.5:
            name_zh = clean_line
        elif not name_en and ascii_ratio >= 0.5 and len(clean_line) > 3:
            name_en = clean_line
        else:
            notes_parts.append(clean_line)

    return name_zh, name_en, " | ".join(notes_parts), course_url


def parse_department_table(soup: BeautifulSoup, dept: dict) -> list[Course]:
    table = _find_main_table(soup)
    if table is None:
        logger.debug(f"  No course table found for {dept['id']}")
        return []

    rows = table.find_all("tr")
    if not rows:
        return []

    # Build column index
    header_cells   = rows[0].find_all(["th", "td"])
    col_keys:      list[str]      = []
    day_col_indices: dict[int, str] = {}

    for idx, cell in enumerate(header_cells):
        raw = _clean(cell.get_text())
        if raw in WEEKDAY_COLS:
            day_col_indices[idx] = raw
            col_keys.append(f"__day_{raw}")
        else:
            col_keys.append(HEADER_KEY_MAP.get(raw, raw))

    has_day_cols = bool(day_col_indices)
    logger.debug(f"  Columns: {col_keys}")
    logger.debug(f"  Day columns: {day_col_indices}")

    courses: list[Course] = []

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells or len(cells) < 3:
            continue

        raw_cell: dict[str, str | Tag] = {}
        for i, key in enumerate(col_keys):
            raw_cell[key] = cells[i] if i < len(cells) else None

        def txt(key: str) -> str:
            val = raw_cell.get(key)
            if val is None:
                return ""
            return _clean(val.get_text() if isinstance(val, Tag) else str(val))

        serial = txt("serial")
        if not serial or not re.search(r"\d", serial):
            continue

        name_cell = raw_cell.get("name")
        if isinstance(name_cell, Tag):
            name_zh, name_en, notes, course_url = _parse_name_cell(name_cell)
        else:
            name_zh, name_en, notes, course_url = txt("name"), "", "", ""

        code    = txt("code").replace(" ", "")
        section = txt("section").strip("*- ")

        instructor = " / ".join(
            p for p in re.split(r"[\s/,、]+", txt("instructor")) if p
        )

        req_raw  = txt("req").strip()
        req_info = REQUIRED_ELECTIVE_MAP.get(req_raw, {"en": req_raw, "zh": req_raw})

        credits_m = re.search(r"\d+\.?\d*", txt("credits"))
        credits   = credits_m.group() if credits_m else txt("credits")
        full_half = txt("fullhalf")
        limit_m   = re.search(r"\d+", txt("limit"))
        limit     = limit_m.group() if limit_m else ""

        prefix_m    = re.match(r"^([A-Za-z]+)", code)
        dept_prefix = prefix_m.group(1).upper() if prefix_m else ""

        schedule_slots: list[ScheduleSlot] = []

        if has_day_cols:
            for col_idx, day_char in day_col_indices.items():
                if col_idx >= len(cells):
                    continue
                cell_text = _clean(cells[col_idx].get_text())
                if not cell_text:
                    continue
                periods, classroom = parse_periods_and_classroom(cell_text)
                if not periods:
                    continue
                bld_en, bld_zh = resolve_building(classroom)
                times = [TIMESLOT_LOOKUP.get(p, {}).get("time", p) for p in periods]
                schedule_slots.append(ScheduleSlot(
                    weekday_zh  = day_char,
                    weekday_en  = WEEKDAY_ZH_TO_EN.get(day_char, day_char),
                    periods     = periods,
                    times       = times,
                    classroom   = classroom,
                    building_en = bld_en,
                    building_zh = bld_zh,
                ))
        else:
            timecls = txt("timecls")
            for m in re.finditer(r"([一二三四五六日])([1-9A-DZa-dz]+)[/／\s]*([\w\-]+)?", timecls):
                day_char   = m.group(1)
                period_str = m.group(2).upper()
                classroom  = (m.group(3) or "").strip()
                periods    = list(period_str)
                bld_en, bld_zh = resolve_building(classroom)
                times = [TIMESLOT_LOOKUP.get(p, {}).get("time", p) for p in periods]
                schedule_slots.append(ScheduleSlot(
                    weekday_zh  = day_char,
                    weekday_en  = WEEKDAY_ZH_TO_EN.get(day_char, day_char),
                    periods     = periods,
                    times       = times,
                    classroom   = classroom,
                    building_en = bld_en,
                    building_zh = bld_zh,
                ))

        courses.append(Course(
            dept_id              = dept["id"],
            dept_name_zh         = dept["name_zh"],
            serial_number        = serial,
            course_code          = code,
            class_section        = section,
            course_name_zh       = name_zh,
            course_name_en       = name_en,
            course_url           = course_url,
            instructor           = instructor,
            credits              = credits,
            required_elective    = req_raw,
            required_elective_en = req_info["en"],
            required_elective_zh = req_info["zh"],
            full_half            = full_half,
            enrollment_limit     = limit,
            notes                = notes,
            schedule_slots       = schedule_slots,
            dept_code_prefix     = dept_prefix,
        ))

    return courses


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, records: Iterator[dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(single_dept: str | None = None, delay: float = DEFAULT_DELAY) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    write_json(OUT_TIMESLOT, TIMESLOT_LOOKUP)
    write_json(OUT_BUILDING, BUILDING_LOOKUP)
    logger.info(f"Lookup tables saved to {DATA_DIR}/")

    session = requests.Session()
    depts   = (
        [{"id": single_dept, "name_zh": f"(debug: {single_dept})"}]
        if single_dept
        else get_departments(session, delay=delay)
    )
    logger.info(f"Scraping {len(depts)} department(s)…\n")

    all_courses:  list[Course] = []
    failed_depts: list[dict]   = []

    for idx, dept in enumerate(depts, 1):
        url = TABLE_URL.format(dept_id=dept["id"])
        logger.info(f"[{idx:>3}/{len(depts)}] {dept['name_zh']}  ({dept['id']})")
        try:
            soup    = fetch(url, session, delay=delay)
            courses = parse_department_table(soup, dept)
            logger.info(f"         → {len(courses)} courses")
            all_courses.extend(courses)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            logger.error(f"         → HTTP {code} — skipping")
            failed_depts.append({**dept, "error": f"HTTP {code}"})
        except Exception as exc:
            logger.error(f"         → {type(exc).__name__}: {exc} — skipping")
            failed_depts.append({**dept, "error": str(exc)})

    if not all_courses:
        logger.error("No courses scraped.")
        sys.exit(1)

    # courses_combined.jsonl is the primary embedding target
    write_jsonl(OUT_COMBINED, (c.to_index_doc_combined() for c in all_courses))
    write_jsonl(OUT_EN,       (c.to_index_doc_en()       for c in all_courses))
    write_jsonl(OUT_ZH,       (c.to_index_doc_zh()       for c in all_courses))
    write_jsonl(OUT_RAW,      (c.to_raw_dict()            for c in all_courses))

    write_json(OUT_REPORT, {
        "total_departments": len(depts),
        "failed_count":      len(failed_depts),
        "total_courses":     len(all_courses),
        "failed_departments": failed_depts,
    })

    print(f"\n{'─'*55}")
    print(f"  Departments : {len(depts) - len(failed_depts)} OK / {len(depts)} total")
    print(f"  Courses     : {len(all_courses)}")
    if failed_depts:
        print(f"  Failed      : {len(failed_depts)}  (see {OUT_REPORT.name})")
    print(f"{'─'*55}")
    print(f"  {'File':<32} {'Size':>12}")
    for p in [OUT_COMBINED, OUT_EN, OUT_ZH, OUT_RAW, OUT_TIMESLOT, OUT_BUILDING, OUT_REPORT]:
        if p.exists():
            print(f"  {p.name:<32} {p.stat().st_size:>10,} bytes")
    print(f"\n  Recommended embedding input: {OUT_COMBINED.name}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NCU Course Scraper")
    ap.add_argument("--dept",  metavar="ID",  help="Scrape one department only (debug)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    main(single_dept=args.dept, delay=args.delay)