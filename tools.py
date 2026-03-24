from __future__ import annotations

import re
import logging
from typing import Optional

from langchain_core.tools import tool
from core import VectorIndex

logger = logging.getLogger(__name__)

MAX_RESULTS      = 5
MAX_CHARS_RESULT = 400   # slightly larger to fit schedule lines
MAX_CHARS_TOTAL  = 3000  # raised so 5 detailed results fit

# ── Department labels ─────────────────────────────────────────────────────────
DEPT_EN_LABEL: dict[str, str] = {
    "體育室":                                    "Physical Education Office",
    "軍訓室":                                    "Military Training Office",
    "學務處-服務學習發展中心":                      "Center for Service-Learning Development",
    "通識教育中心":                               "Center for General Education",
    "語言中心":                                   "Language Center",
    "總教學中心":                                 "Center for Teaching and Learning Development",
    "學務處-職涯發展中心":                         "Career Development Center",
    "臺灣大專院校人工智慧學程聯盟":                 "Taiwan Alliance of AI Education in Universities",
    "核心通識課程":                               "Core General Education Courses",
    "師資培育中心":                               "Center for Teacher Education",
    "遙測科技碩士學位學程":                        "Master's Program in Remote Sensing Technology",
    "環境科技博士學位學程(台灣聯合大學系統)":        "Ph.D. Program in Environmental Technology (UST)",
    "中國文學系":                                 "Department of Chinese Literature",
    "英美語文學系":                               "Department of English Language and Literature",
    "法國語文學系":                               "Department of French Language and Literature",
    "文學院學士班":                               "College of Liberal Arts — Bachelor's Program",
    "哲學研究所碩士班":                           "Graduate Institute of Philosophy",
    "歷史研究所碩士班":                           "Graduate Institute of History",
    "藝術學研究所碩士班":                         "Graduate Institute of Art Studies",
    "學習與教學研究所碩士班":                      "Graduate Institute of Learning and Instruction",
    "亞際文化研究國際碩士學位學程":                 "International Master's in Inter-Asia Cultural Studies",
    "理學院":                                     "College of Science",
    "數學系":                                     "Department of Mathematics",
    "物理學系":                                   "Department of Physics",
    "化學學系":                                   "Department of Chemistry",
    "光電科學與工程學系":                          "Department of Optics and Photonics",
    "理學院學士班":                               "College of Science — Bachelor's Program",
    "統計研究所碩士班":                           "Graduate Institute of Statistics",
    "天文研究所碩士班":                           "Graduate Institute of Astronomy",
    "工學院":                                     "College of Engineering",
    "土木工程學系":                               "Department of Civil Engineering",
    "機械工程學系":                               "Department of Mechanical Engineering",
    "化學工程與材料工程學系":                      "Department of Chemical and Materials Engineering",
    "工學院學士班":                               "College of Engineering — Bachelor's Program",
    "營建管理研究所碩士班":                        "Graduate Institute of Construction Engineering and Management",
    "環境工程研究所碩士班":                        "Graduate Institute of Environmental Engineering",
    "能源工程研究所碩士班":                        "Graduate Institute of Energy Engineering",
    "材料科學與工程研究所碩士班":                  "Graduate Institute of Materials Science and Engineering",
    "土木工程學系營建管理碩士班":                  "Civil Engineering — Construction Management Master's",
    "管理學院":                                   "College of Management",
    "企業管理學系":                               "Department of Business Administration",
    "資訊管理學系":                               "Department of Information Management",
    "財務金融學系":                               "Department of Finance",
    "經濟學系":                                   "Department of Economics",
    "產業經濟研究所碩士班":                        "Graduate Institute of Industrial Economics",
    "工業管理研究所碩士班":                        "Graduate Institute of Industrial Management",
    "人力資源管理研究所碩士班":                    "Graduate Institute of Human Resource Management",
    "會計研究所碩士班":                           "Graduate Institute of Accounting",
    "國際經營管理碩士學位學程":                    "International MBA (IMBA)",
    "管理學院高階主管企管碩士班":                  "Executive MBA (EMBA)",
    "碩士在職現役軍人營區專班":                    "In-service Master's for Military Personnel",
    "電機工程學系":                               "Department of Electrical Engineering",
    "資訊工程學系":                               "Department of Computer Science and Engineering",
    "通訊工程學系":                               "Department of Communication Engineering",
    "資訊電機學院學士班":                          "College of EECS — Bachelor's Program",
    "網路學習科技研究所碩士班":                    "Graduate Institute of Network Learning Technology",
    "人工智慧國際碩士學位學程":                    "International Master's in Artificial Intelligence",
    "大氣科學學系":                               "Department of Atmospheric Sciences",
    "地球科學學系":                               "Department of Earth Sciences",
    "地球科學學院學士班":                          "College of Earth Sciences — Bachelor's Program",
    "太空科學與工程學系":                          "Department of Space Science and Engineering",
    "應用地質研究所碩士班":                        "Graduate Institute of Applied Geology",
    "水文與海洋科學研究所碩士班":                  "Graduate Institute of Hydrological and Oceanic Sciences",
    "地球系統科學國際研究生博士學位學程":           "International Ph.D. in Earth System Science",
    "客家語文暨社會科學學系":                      "Department of Hakka Language and Social Sciences",
    "法律與政府研究所碩士班":                      "Graduate Institute of Law and Government",
    "客家語文暨社會科學學系客家社會及政策碩士班":   "Master's in Hakka Society and Policy",
    "客家語文暨社會科學學系客家語文碩士班":         "Master's in Hakka Language",
    "客家語文暨社會科學學系客家研究碩士在職專班":   "In-service Master's in Hakka Studies",
    "客家語文暨社會科學學系客家研究博士班":         "Ph.D. in Hakka Studies",
    "生命科學系":                                 "Department of Life Science",
    "生醫科學與工程學系":                          "Department of Biomedical Sciences and Engineering",
    "認知神經科學研究所碩士班":                    "Graduate Institute of Cognitive Neuroscience",
    "跨領域轉譯醫學研究所碩士班":                  "Graduate Institute of Translational Medicine",
    "生醫科學與工程學系系統生物與生物資訊碩士班":   "Master's in Systems Biology and Bioinformatics",
    "生醫科學與工程學系生物醫學工程碩士班":         "Master's in Biomedical Engineering",
    "生醫科學與工程學系跨領域轉譯醫學博士班":      "Ph.D. in Interdisciplinary Translational Medicine",
    "跨領域神經科學國際研究生博士學位學程":         "International Ph.D. in Interdisciplinary Neuroscience",
    "永續與綠能科技研究學院":                      "Graduate College of Sustainability and Green Energy",
    "永續去碳科技碩士學位學程":                    "Master's in Sustainable Decarbonization Technology",
    "永續綠能科技碩士學位學程":                    "Master's in Sustainable Green Energy Technology",
    "永續領導力碩士學位學程":                      "Master's in Sustainable Leadership",
}

# Only full/meaningful names — no abbreviations
DEPT_MAP: dict[str, str] = {
    # EECS
    "computer science":                            "資訊工程學系",
    "computer science and engineering":            "資訊工程學系",
    "資訊工程":                                     "資訊工程學系",
    "資訊工程學系":                                  "資訊工程學系",
    "electrical engineering":                      "電機工程學系",
    "電機工程":                                     "電機工程學系",
    "電機工程學系":                                  "電機工程學系",
    "communication engineering":                   "通訊工程學系",
    "通訊工程":                                     "通訊工程學系",
    "通訊工程學系":                                  "通訊工程學系",
    "information management":                      "資訊管理學系",
    "資訊管理":                                     "資訊管理學系",
    "資訊管理學系":                                  "資訊管理學系",
    "optics and photonics":                        "光電科學與工程學系",
    "photonics":                                   "光電科學與工程學系",
    "光電科學與工程":                                "光電科學與工程學系",
    "光電科學與工程學系":                             "光電科學與工程學系",
    "network learning technology":                 "網路學習科技研究所碩士班",
    "graduate institute of network learning technology": "網路學習科技研究所碩士班",
    "網路學習科技":                                  "網路學習科技研究所碩士班",
    "網路學習科技研究所碩士班":                       "網路學習科技研究所碩士班",
    "artificial intelligence":                     "人工智慧國際碩士學位學程",
    "international master artificial intelligence": "人工智慧國際碩士學位學程",
    "人工智慧":                                     "人工智慧國際碩士學位學程",
    "人工智慧國際碩士學位學程":                       "人工智慧國際碩士學位學程",
    # Engineering
    "mechanical engineering":                      "機械工程學系",
    "機械工程":                                     "機械工程學系",
    "機械工程學系":                                  "機械工程學系",
    "civil engineering":                           "土木工程學系",
    "土木工程":                                     "土木工程學系",
    "土木工程學系":                                  "土木工程學系",
    "chemical and materials engineering":          "化學工程與材料工程學系",
    "chemical engineering":                        "化學工程與材料工程學系",
    "materials engineering":                       "化學工程與材料工程學系",
    "化學工程與材料工程":                             "化學工程與材料工程學系",
    "化學工程與材料工程學系":                          "化學工程與材料工程學系",
    # Science
    "mathematics":                                 "數學系",
    "數學":                                         "數學系",
    "數學系":                                       "數學系",
    "physics":                                     "物理學系",
    "物理":                                         "物理學系",
    "物理學系":                                     "物理學系",
    "chemistry":                                   "化學學系",
    "化學":                                         "化學學系",
    "化學學系":                                     "化學學系",
    "life science":                                "生命科學系",
    "life sciences":                               "生命科學系",
    "biology":                                     "生命科學系",
    "生命科學":                                     "生命科學系",
    "生命科學系":                                   "生命科學系",
    "earth sciences":                              "地球科學學系",
    "earth science":                               "地球科學學系",
    "地球科學":                                     "地球科學學系",
    "地球科學學系":                                  "地球科學學系",
    "atmospheric sciences":                        "大氣科學學系",
    "atmospheric science":                         "大氣科學學系",
    "meteorology":                                 "大氣科學學系",
    "大氣科學":                                     "大氣科學學系",
    "大氣科學學系":                                  "大氣科學學系",
    "space science and engineering":               "太空科學與工程學系",
    "space science":                               "太空科學與工程學系",
    "太空科學與工程":                                "太空科學與工程學系",
    "太空科學與工程學系":                             "太空科學與工程學系",
    "statistics":                                  "統計研究所碩士班",
    "graduate institute of statistics":            "統計研究所碩士班",
    "統計":                                         "統計研究所碩士班",
    "astronomy":                                   "天文研究所碩士班",
    "graduate institute of astronomy":             "天文研究所碩士班",
    "天文":                                         "天文研究所碩士班",
    # Management
    "business administration":                     "企業管理學系",
    "企業管理":                                     "企業管理學系",
    "企業管理學系":                                  "企業管理學系",
    "finance":                                     "財務金融學系",
    "financial management":                        "財務金融學系",
    "財務金融":                                     "財務金融學系",
    "財務金融學系":                                  "財務金融學系",
    "economics":                                   "經濟學系",
    "經濟":                                         "經濟學系",
    "經濟學系":                                     "經濟學系",
    "industrial economics":                        "產業經濟研究所碩士班",
    "產業經濟":                                     "產業經濟研究所碩士班",
    "industrial management":                       "工業管理研究所碩士班",
    "工業管理":                                     "工業管理研究所碩士班",
    "human resource management":                   "人力資源管理研究所碩士班",
    "人力資源管理":                                  "人力資源管理研究所碩士班",
    "accounting":                                  "會計研究所碩士班",
    "會計":                                         "會計研究所碩士班",
    "international mba":                           "國際經營管理碩士學位學程",
    "國際經營管理":                                  "國際經營管理碩士學位學程",
    # Liberal Arts
    "chinese literature":                          "中國文學系",
    "中國文學":                                     "中國文學系",
    "中國文學系":                                   "中國文學系",
    "english language and literature":             "英美語文學系",
    "english literature":                          "英美語文學系",
    "英美語文":                                     "英美語文學系",
    "英美語文學系":                                  "英美語文學系",
    "french language and literature":              "法國語文學系",
    "french":                                      "法國語文學系",
    "法國語文":                                     "法國語文學系",
    "法國語文學系":                                  "法國語文學系",
    "hakka language and social sciences":          "客家語文暨社會科學學系",
    "hakka":                                       "客家語文暨社會科學學系",
    "客家語文暨社會科學":                             "客家語文暨社會科學學系",
    "philosophy":                                  "哲學研究所碩士班",
    "哲學":                                         "哲學研究所碩士班",
    "history":                                     "歷史研究所碩士班",
    "歷史":                                         "歷史研究所碩士班",
    "learning and instruction":                    "學習與教學研究所碩士班",
    "學習與教學":                                   "學習與教學研究所碩士班",
    # Earth Sciences
    "applied geology":                             "應用地質研究所碩士班",
    "應用地質":                                     "應用地質研究所碩士班",
    "hydrology":                                   "水文與海洋科學研究所碩士班",
    "oceanography":                                "水文與海洋科學研究所碩士班",
    "水文與海洋科學":                                "水文與海洋科學研究所碩士班",
    # Health
    "biomedical sciences and engineering":         "生醫科學與工程學系",
    "biomedical":                                  "生醫科學與工程學系",
    "生醫科學與工程":                                "生醫科學與工程學系",
    "生醫科學與工程學系":                             "生醫科學與工程學系",
    "cognitive neuroscience":                      "認知神經科學研究所碩士班",
    "neuroscience":                                "認知神經科學研究所碩士班",
    "認知神經科學":                                  "認知神經科學研究所碩士班",
    "translational medicine":                      "跨領域轉譯醫學研究所碩士班",
    "轉譯醫學":                                     "跨領域轉譯醫學研究所碩士班",
    # Sustainability
    "sustainability and green energy":             "永續與綠能科技研究學院",
    "green energy":                                "永續與綠能科技研究學院",
    "永續與綠能科技":                                "永續與綠能科技研究學院",
    # Centres
    "physical education":                          "體育室",
    "體育":                                         "體育室",
    "language center":                             "語言中心",
    "語言中心":                                     "語言中心",
    "general education":                           "通識教育中心",
    "center for general education":                "通識教育中心",
    "通識教育":                                     "通識教育中心",
    "通識教育中心":                                  "通識教育中心",
    "teacher education":                           "師資培育中心",
    "師資培育":                                     "師資培育中心",
    "service learning":                            "學務處-服務學習發展中心",
    "服務學習":                                     "學務處-服務學習發展中心",
    "career development":                          "學務處-職涯發展中心",
    "職涯發展":                                     "學務處-職涯發展中心",
    "military training":                           "軍訓室",
    "軍訓":                                         "軍訓室",
}

# ── EECS scope ────────────────────────────────────────────────────────────────
EECS_DEPTS_ZH: frozenset[str] = frozenset({
    "電機工程學系",
    "資訊工程學系",
    "通訊工程學系",
    "資訊電機學院學士班",
    "網路學習科技研究所碩士班",
    "人工智慧國際碩士學位學程",
})

VALID_DAYS = {
    "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday",
}

WEEKDAY_NORM: dict[str, str] = {
    "mon": "monday",    "monday": "monday",    "一": "monday",    "星期一": "monday",
    "tue": "tuesday",   "tuesday": "tuesday",  "二": "tuesday",   "星期二": "tuesday",
    "wed": "wednesday", "wednesday": "wednesday","三": "wednesday","星期三": "wednesday",
    "thu": "thursday",  "thursday": "thursday", "四": "thursday",  "星期四": "thursday",
    "fri": "friday",    "friday": "friday",     "五": "friday",    "星期五": "friday",
    "sat": "saturday",  "saturday": "saturday", "六": "saturday",  "星期六": "saturday",
    "sun": "sunday",    "sunday": "sunday",     "日": "sunday",    "星期日": "sunday",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_index(index: Optional[VectorIndex]) -> Optional[str]:
    if index is None:
        return "Index not loaded. Run: python index_builder.py"
    if not index.documents:
        return "Index empty. Run: python index_builder.py"
    return None


def _resolve_dept(name: str) -> tuple[str | None, str]:
    key = name.strip().lower()
    if key in DEPT_MAP:
        zh = DEPT_MAP[key]
        return zh, DEPT_EN_LABEL.get(zh, zh)
    for alias, zh in DEPT_MAP.items():
        if key in alias or alias in key:
            return zh, DEPT_EN_LABEL.get(zh, zh)
    if any("\u4e00" <= c <= "\u9fff" for c in name):
        zh = name.strip()
        return zh, DEPT_EN_LABEL.get(zh, zh)
    return None, name


def _find_ambiguous_depts(
    query: str,
    index: VectorIndex,
    restrict_to: Optional[frozenset[str]] = None,
) -> list[tuple[str, str, int]]:
    q       = query.strip().lower()
    seen    = set()
    matches = []

    def add(zh: str) -> None:
        if zh in seen:
            return
        if restrict_to is not None and zh not in restrict_to:
            return
        count = sum(1 for d in index.documents if d.get("dept_name_zh") == zh)
        if count > 0:
            seen.add(zh)
            matches.append((DEPT_EN_LABEL.get(zh, zh), zh, count))

    for alias, zh in DEPT_MAP.items():
        if q in alias.lower() or alias.lower() in q:
            add(zh)
    for zh, en in DEPT_EN_LABEL.items():
        if q in en.lower() or q in zh.lower():
            add(zh)

    return sorted(matches, key=lambda x: x[2], reverse=True)


def _find_similar_courses(
    query: str,
    index: VectorIndex,
) -> list[tuple[str, str, str]]:
    q       = query.strip().lower()
    matches = []
    seen    = set()
    for d in index.documents:
        name_en = (d.get("course_name_en") or d.get("course_name") or "").lower()
        name_zh = (d.get("course_name_zh") or "").lower()
        code    = d.get("course_code", "")
        dept    = d.get("dept_name_zh", "")
        key     = f"{code}-{dept}"
        if key in seen:
            continue
        if q in name_en or q in name_zh or name_en in q or name_zh in q:
            seen.add(key)
            display = d.get("course_name_en") or d.get("course_name") or name_zh
            matches.append((display, code, dept))
    return matches


def _parse_credits_val(doc: dict) -> int:
    cr = doc.get("credits", "")
    if not cr:
        m = re.search(r"Credits:\s*(\d+)|學分數：(\d+)", doc.get("text", ""))
        cr = (m.group(1) or m.group(2)) if m else "0"
    try:
        return int(cr)
    except (ValueError, TypeError):
        return 0


def _parse_type(doc: dict) -> str:
    m = re.search(r"Type:\s*(\w+)|修課性質：(\S+)", doc.get("text", ""))
    return (m.group(1) or m.group(2)) if m else ""


def _compress(text: str) -> str:
    DROP = ("分發條件", "備註：", "課程詳細頁面", "URL:", "Notes:")
    lines, cr_line = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line or any(line.startswith(p) for p in DROP):
            continue
        if line.startswith(("Credits:", "學分數：")):
            cr_line = line
        else:
            lines.append(line)
    if cr_line:
        lines.insert(1, cr_line)
    result = "\n".join(lines)
    return result[:MAX_CHARS_RESULT] + "…" if len(result) > MAX_CHARS_RESULT else result


def _format_results(texts: list[str], label: str) -> str:
    """Format up to MAX_RESULTS (5) deduplicated results."""
    seen, unique = set(), []
    for t in texts:
        k = t.strip()
        if k not in seen:
            seen.add(k)
            unique.append(_compress(k))
    if not unique:
        return f"No courses found for '{label}'."
    parts, total = [], 0
    for i, r in enumerate(unique[:MAX_RESULTS]):
        entry = f"[{i+1}] {r}"
        total += len(entry)
        if total > MAX_CHARS_TOTAL:
            parts.append(f"[{i+1}] …(truncated)")
            break
        parts.append(entry)
    header = f"Found {len(unique)} result(s) for '{label}' (showing {len(parts)}):\n\n"
    return header + "\n\n".join(parts)


def _dept_docs(index: VectorIndex, dept_zh: str) -> list[dict]:
    return [d for d in index.documents if d.get("dept_name_zh", "") == dept_zh]


def _eecs_docs(index: VectorIndex) -> list[dict]:
    return [d for d in index.documents if d.get("dept_name_zh", "") in EECS_DEPTS_ZH]


def _format_graph_results(docs: list[dict], query_desc: str, max_results: int = MAX_RESULTS) -> str:
    """
    Format up to max_results courses with full detail:
    code, name, dept, credits, instructor, schedule (days + periods + times + rooms).
    """
    if not docs:
        return f"No courses found matching: {query_desc}"

    lines = []
    for i, d in enumerate(docs[:max_results]):
        code  = d.get("course_code", "?")
        name  = d.get("course_name_en") or d.get("course_name") or "?"
        dept  = d.get("dept_name_zh", "")
        cr    = d.get("credits", "?")
        inst  = d.get("instructor", "")
        ctype = _parse_type(d)

        # ── Header line ────────────────────────────────────────────────────
        header = f"[{i+1}] [{code}] {name} — {cr} credits"
        if ctype:
            header += f" ({ctype})"
        parts = []
        if dept:  parts.append(f"Dept: {dept}")
        if inst:  parts.append(f"Instructor: {inst}")
        if parts:
            header += "\n   " + " | ".join(parts)

        # ── Schedule lines — one per slot ──────────────────────────────────
        # Try to reconstruct from metadata first; fall back to text parsing.
        schedule_lines = _extract_schedule_lines(d)
        if schedule_lines:
            header += "\n   " + "\n   ".join(schedule_lines)

        lines.append(header)

    total = len(docs)
    shown = min(total, max_results)
    header_str = f"Found {total} course(s) matching '{query_desc}' (showing {shown}):\n\n"
    return header_str + "\n\n".join(lines)


def _extract_schedule_lines(doc: dict) -> list[str]:
    """
    Build human-readable schedule lines for a course document.

    Priority:
      1. Structured 'schedule' list (new field added by scraper fix)
      2. Flat weekdays/periods/classrooms metadata (old format, merged view)
      3. Regex parse of prose text (last resort)

    Returns lines like:
      📅 Monday | Periods: 3,4 (10:00–11:50) | Room: E6-A207 (Engineering Building 7)
    """
    # ── 1. Structured per-slot schedule list (preferred) ──────────────────
    schedule = doc.get("schedule", [])
    if schedule:
        lines = []
        for slot in schedule:
            day      = (slot.get("weekday") or "").title()
            periods  = slot.get("periods", [])
            times    = slot.get("times", [])
            room     = slot.get("classroom", "")
            bld      = slot.get("building", "")

            if not day or not periods:
                continue

            period_str = ", ".join(periods)
            time_str   = ""
            if times:
                # Build range from first start to last end
                try:
                    start = times[0].split("-")[0].strip()
                    end   = times[-1].split("-")[1].strip()
                    time_str = f" ({start}–{end})"
                except (IndexError, AttributeError):
                    time_str = f" ({', '.join(times)})"

            line = f"📅 {day} | Periods: {period_str}{time_str}"
            if room:
                line += f" | Room: {room}"
                if bld:
                    line += f" ({bld})"
            elif bld:
                line += f" | Building: {bld}"
            lines.append(line)
        if lines:
            return lines

    # ── 2. Flat metadata fallback (weekdays + periods merged, old format) ──
    weekdays   = doc.get("weekdays", [])
    periods    = doc.get("periods", [])
    classrooms = doc.get("classrooms", [])
    buildings  = doc.get("buildings", [])

    if weekdays and periods:
        day_str    = ", ".join(w.title() for w in weekdays)
        period_str = ", ".join(sorted(periods))
        room_str   = ", ".join(classrooms) if classrooms else ""
        bld_str    = ", ".join(buildings)  if buildings  else ""
        line = f"📅 {day_str} | Periods: {period_str}"
        if room_str:
            line += f" | Room: {room_str}"
            if bld_str:
                line += f" ({bld_str})"
        elif bld_str:
            line += f" | Building: {bld_str}"
        return [line]

    # ── 3. Parse prose text as last resort ─────────────────────────────────
    text    = doc.get("text", "")
    results = []

    # English prose: "Monday periods 3 and 4 (10:00–11:50) in room E6-A207 (Eng Bldg 7)"
    for m in re.finditer(
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        r"\s+periods?\s+([\d,\s\w]+?)"
        r"\s*\(([^)]+)\)"
        r"(?:\s+in room\s+([\w\-]+))?"
        r"(?:\s+\(([^)]+)\))?",
        text, re.IGNORECASE
    ):
        day, pds, time_range, room, bld = m.groups()
        line = f"📅 {day.title()} | Periods: {pds.strip()} ({time_range.strip()})"
        if room:
            line += f" | Room: {room.strip()}"
            if bld:
                line += f" ({bld.strip()})"
        results.append(line)

    if results:
        return results

    # Chinese prose: "每週三第3、4節（10:00–11:50）在E6-A207教室（工程七館）"
    ZH_DAY = {"一":"Mon","二":"Tue","三":"Wed","四":"Thu","五":"Fri","六":"Sat","日":"Sun"}
    for m in re.finditer(
        r"每週([一二三四五六日])第([\d、A-DZa-dz]+)節"
        r"[（(]([^）)]+)[）)]"
        r"(?:在([\w\-]+)教室)?(?:[（(]([^）)]+)[）)])?",
        text
    ):
        day_zh, pds, time_range, room, bld = m.groups()
        day_en = ZH_DAY.get(day_zh, day_zh)
        line = f"📅 {day_en} | Periods: {pds} ({time_range})"
        if room:
            line += f" | Room: {room}"
            if bld:
                line += f" ({bld})"
        results.append(line)

    return results


def _apply_filters(
    docs:        list[dict],
    department:  Optional[str] = None,
    building:    Optional[str] = None,
    weekday:     Optional[str] = None,
    period:      Optional[str] = None,
    credits:     Optional[int] = None,
    instructor:  Optional[str] = None,
    course_name: Optional[str] = None,
    req_type:    Optional[str] = None,
) -> list[dict]:
    result = docs
    if department:
        dept_zh, _ = _resolve_dept(department)
        if dept_zh:
            result = [d for d in result if d.get("dept_name_zh") == dept_zh]
    if building:
        b = building.strip().lower()
        result = [
            d for d in result
            if any(b in bld.lower() for bld in d.get("buildings", []))
            or any(b in cls.lower() for cls in d.get("classrooms", []))
        ]
    if weekday:
        w = WEEKDAY_NORM.get(weekday.strip().lower(), weekday.strip().lower())
        result = [d for d in result if w in [x.lower() for x in d.get("weekdays", [])]]
    if period:
        p = period.strip().upper()
        result = [d for d in result if p in [x.upper() for x in d.get("periods", [])]]
    if credits is not None:
        result = [d for d in result if _parse_credits_val(d) == credits]
    if instructor:
        parts = [p.strip() for p in re.split(r"[,\s\-]+", instructor.strip().lower()) if p.strip()]
        result = [
            d for d in result
            if all(p in d.get("instructor", "").lower() for p in parts)
        ]
    if course_name:
        cn = course_name.strip().lower()
        result = [
            d for d in result
            if cn in (d.get("course_name", "") or "").lower()
            or cn in (d.get("course_name_zh", "") or "").lower()
            or cn in (d.get("course_code", "") or "").lower()
            or cn in (d.get("text", "") or "").lower()
        ]
    if req_type:
        rt = req_type.strip().lower()
        result = [d for d in result if rt in d.get("text", "").lower()]
    return result


def _build_course_plan(courses: list[dict], target_credits: int, display: str) -> str:
    total_available = sum(c["credits"] for c in courses)
    if total_available < target_credits:
        return (
            f"{display} only has {total_available} total credits across "
            f"{len(courses)} courses — cannot reach {target_credits}."
        )

    def is_required(c: dict) -> bool:
        return "required" in c["type"].lower() or "必" in c["type"]

    required  = sorted([c for c in courses if is_required(c)],     key=lambda x: x["credits"], reverse=True)
    electives = sorted([c for c in courses if not is_required(c)], key=lambda x: x["credits"], reverse=True)
    ordered   = required + electives

    selected, total = [], 0
    for c in ordered:
        if total >= target_credits:
            break
        if total + c["credits"] <= target_credits:
            selected.append(c)
            total += c["credits"]

    gap = target_credits - total
    if gap > 0:
        for c in ordered:
            if c not in selected and c["credits"] == gap:
                selected.append(c)
                total += c["credits"]
                break

    if not selected:
        return f"Could not find a valid combination totalling {target_credits} credits."

    lines, running = [], 0
    for c in selected:
        running += c["credits"]
        tag = f" ({c['type']})" if c["type"] else ""
        lines.append(f"[{c['code']}] {c['name']} — {c['credits']} cr{tag}  → running: {running}")

    status = "exactly" if total == target_credits else f"closest: {total}"
    sep    = "─" * 44
    return (
        f"{display}\nTarget: {target_credits} credits ({status})\n{sep}\n"
        + "\n".join(lines)
        + f"\n{sep}\nTotal: {total} / {target_credits} credits"
    )


def _collect_courses(docs: list[dict]) -> list[dict]:
    courses = []
    for d in docs:
        cr    = _parse_credits_val(d)
        code  = d.get("course_code", "?")
        name  = d.get("course_name_en") or d.get("course_name") or "?"
        ctype = _parse_type(d)
        if cr == 0:
            continue
        courses.append({"code": code, "name": name, "credits": cr, "type": ctype})
    return courses


# ── Tool factory ──────────────────────────────────────────────────────────────

def create_tools(index: Optional[VectorIndex]) -> list:

    @tool
    def clarify(question: str) -> str:
        """
        Ask the user to clarify when their input is ambiguous.

        Args:
            question: The clarifying question to present to the user,
                      including numbered options they can choose from.
        """
        return f"[CLARIFICATION NEEDED]\n{question}"

    # ── Ambiguity detection ───────────────────────────────────────────────────

    @tool
    def detect_ambiguity(query: str) -> str:
        """
        Check if a query could mean multiple departments OR multiple courses.
        Call this when unsure before calling any search/listing tool.

        Args:
            query: The user's input that might be ambiguous.
        """
        if err := _check_index(index):
            return err

        q = query.strip()
        dept_matches   = _find_ambiguous_depts(q, index)
        course_matches = _find_similar_courses(q, index)

        if len(dept_matches) <= 1 and len(course_matches) <= 1:
            if dept_matches:
                en, zh, count = dept_matches[0]
                return f"UNAMBIGUOUS_DEPT: {zh} ({en}) — {count} courses"
            if course_matches:
                name, code, dept = course_matches[0]
                return f"UNAMBIGUOUS_COURSE: [{code}] {name} ({dept})"
            return f"NO_MATCH: '{q}' did not match any department or course name."

        lines = [f"AMBIGUOUS: '{q}' could mean multiple things.\n"]
        if len(dept_matches) > 1:
            lines.append(f"As a DEPARTMENT ({len(dept_matches)} matches):")
            for i, (en, zh, count) in enumerate(dept_matches[:5], 1):
                lines.append(f"  {i}. {en} ({zh}) — {count} courses")
        if len(course_matches) > 1:
            lines.append(f"\nAs a COURSE NAME ({len(course_matches)} matches):")
            for i, (name, code, dept) in enumerate(course_matches[:5], 1):
                dept_en = DEPT_EN_LABEL.get(dept, dept)
                lines.append(f"  {i}. [{code}] {name} — {dept_en}")
        if dept_matches and course_matches:
            lines.append(f"\nNote: '{q}' could refer to EITHER a department OR a course.")
        lines.append("\nPlease ask the user to clarify.")
        return "\n".join(lines)

    @tool
    def detect_ambiguity_eecs(query: str) -> str:
        """
        Check if a query matches multiple EECS departments or courses.

        Args:
            query: The user's input that might be ambiguous within EECS.
        """
        if err := _check_index(index):
            return err

        q = query.strip()
        dept_matches   = _find_ambiguous_depts(q, index, restrict_to=EECS_DEPTS_ZH)
        course_matches = [
            (n, c, d) for n, c, d in _find_similar_courses(q, index)
            if d in EECS_DEPTS_ZH
        ]

        if len(dept_matches) <= 1 and len(course_matches) <= 1:
            if dept_matches:
                en, zh, count = dept_matches[0]
                return f"UNAMBIGUOUS_DEPT: {zh} ({en}) — {count} courses"
            if course_matches:
                name, code, dept = course_matches[0]
                return f"UNAMBIGUOUS_COURSE: [{code}] {name} ({dept})"
            return f"NO_MATCH: '{q}' not found in EECS departments or courses."

        lines = [f"AMBIGUOUS: '{q}' could mean multiple things in EECS.\n"]
        if len(dept_matches) > 1:
            lines.append(f"As a DEPARTMENT ({len(dept_matches)} matches):")
            for i, (en, zh, count) in enumerate(dept_matches[:5], 1):
                lines.append(f"  {i}. {en} ({zh}) — {count} courses")
        if len(course_matches) > 1:
            lines.append(f"\nAs a COURSE NAME ({len(course_matches)} matches):")
            for i, (name, code, dept) in enumerate(course_matches[:5], 1):
                dept_en = DEPT_EN_LABEL.get(dept, dept)
                lines.append(f"  {i}. [{code}] {name} — {dept_en}")
        if dept_matches and course_matches:
            lines.append(f"\nNote: could be a department OR a course.")
        lines.append("\nPlease ask the user to clarify.")
        return "\n".join(lines)

    # ── Content search ────────────────────────────────────────────────────────

    @tool
    def search_courses_by_content(query: str) -> str:
        """
        Search all NCU courses by topic, keyword, or professor name.

        Args:
            query: Natural language search query.
        """
        if err := _check_index(index): return err
        if not query.strip(): return "Please provide a non-empty query."
        results = index.search(query.strip(), top_k=MAX_RESULTS)
        return _format_results([r.text for r in results], query)

    @tool
    def search_eecs_courses_by_content(query: str) -> str:
        """
        Search EECS courses (CS, EE, CE, NLT, AI) by topic, keyword, or professor name.

        Args:
            query: Natural language search query.
        """
        if err := _check_index(index): return err
        if not query.strip(): return "Please provide a non-empty query."
        eecs_texts = {d.get("text") for d in _eecs_docs(index)}
        results    = [
            r for r in index.search(query.strip(), top_k=MAX_RESULTS * 2)
            if r.text in eecs_texts
        ][:MAX_RESULTS]
        if not results:
            return f"No EECS courses found for '{query}'."
        return _format_results([r.text for r in results], query)

    # ── Department search ─────────────────────────────────────────────────────

    @tool
    def search_courses_by_department(
        department: str,
        keyword:    Optional[str] = None,
    ) -> str:
        """
        List up to 5 courses from a department with optional keyword filter.

        Args:
            department: Full department name in English or Chinese.
            keyword:    Optional filter keyword (null if not needed).
        """
        if err := _check_index(index): return err
        dept_zh, display = _resolve_dept(department)
        if dept_zh is None:
            ambiguous = _find_ambiguous_depts(department, index)
            if ambiguous:
                lines = [f"Department '{department}' not found exactly. Did you mean:"]
                for en, zh, count in ambiguous[:5]:
                    lines.append(f"  • {en} ({zh}) — {count} courses")
                return "\n".join(lines)
            return f"Department '{department}' not found."
        docs = _dept_docs(index, dept_zh)
        if keyword:
            kw   = keyword.strip().lower()
            docs = [d for d in docs if kw in d.get("text", "").lower()]
            label = f"{display} / {keyword}"
        else:
            label = display
        return _format_results([d["text"] for d in docs], label)

    # ── Full course listing ───────────────────────────────────────────────────

    def _all_courses_for_dept(department: str, docs: list[dict], display: str) -> str:
        if not docs:
            return f"No courses found for {display}."
        lines, skipped = [], 0
        for d in docs:
            cr    = _parse_credits_val(d)
            code  = d.get("course_code", "?")
            name  = d.get("course_name_en") or d.get("course_name") or "?"
            ctype = _parse_type(d)
            if cr == 0:
                skipped += 1
                continue
            line = f"[{code}] {name} — {cr} cr"
            if ctype:
                line += f" ({ctype})"
            lines.append((cr, line))
        lines.sort(key=lambda x: x[0], reverse=True)
        body   = "\n".join(l for _, l in lines)
        note   = f" ({skipped} zero-credit excluded)" if skipped else ""
        header = f"{display} — {len(lines)} courses{note}:\n"
        if len(body) > 2500:
            kept, total_len = [], 0
            for _, line in lines:
                if total_len + len(line) + 1 > 2500:
                    break
                kept.append(line)
                total_len += len(line) + 1
            body = "\n".join(kept) + f"\n…({len(lines)-len(kept)} more)"
        return header + body

    @tool
    def get_all_courses_by_department(department: str) -> str:
        """
        Return ALL courses from any NCU department as a compact one-line-per-course list.

        Args:
            department: Full department name in English or Chinese.
        """
        if err := _check_index(index): return err
        dept_zh, display = _resolve_dept(department)
        if dept_zh is None:
            ambiguous = _find_ambiguous_depts(department, index)
            if len(ambiguous) > 1:
                lines = [f"'{department}' matches multiple departments. Please specify:"]
                for i, (en, zh, count) in enumerate(ambiguous, 1):
                    lines.append(f"  {i}. {en} ({zh}) — {count} courses")
                return "\n".join(lines)
            elif len(ambiguous) == 1:
                dept_zh, display = ambiguous[0][1], ambiguous[0][0]
            else:
                return f"Department '{department}' not found."
        return _all_courses_for_dept(department, _dept_docs(index, dept_zh), display)

    @tool
    def get_all_eecs_courses_by_department(department: str) -> str:
        """
        Return ALL courses from an EECS department (CS, EE, CE, NLT, AI).

        Args:
            department: Full department name in English or Chinese.
        """
        if err := _check_index(index): return err
        dept_zh, display = _resolve_dept(department)
        if dept_zh is None:
            ambiguous = _find_ambiguous_depts(department, index, restrict_to=EECS_DEPTS_ZH)
            if len(ambiguous) > 1:
                lines = [f"'{department}' matches multiple EECS departments:"]
                for i, (en, zh, count) in enumerate(ambiguous, 1):
                    lines.append(f"  {i}. {en} ({zh}) — {count} courses")
                return "\n".join(lines)
            elif len(ambiguous) == 1:
                dept_zh, display = ambiguous[0][1], ambiguous[0][0]
            else:
                return f"Department '{department}' not found in EECS."
        if dept_zh not in EECS_DEPTS_ZH:
            return f"'{display}' is outside EECS scope. I only cover: CS, EE, CE, NLT, and AI."
        return _all_courses_for_dept(department, _dept_docs(index, dept_zh), display)

    # ── Credit planning ───────────────────────────────────────────────────────

    @tool
    def plan_courses_by_credits(department: str, target_credits: int) -> str:
        """
        Select courses from any NCU department whose credits sum to target_credits.

        Args:
            department:     Full department name in English or Chinese.
            target_credits: Desired total credits e.g. 24.
        """
        if err := _check_index(index): return err
        dept_zh, display = _resolve_dept(department)
        if dept_zh is None:
            ambiguous = _find_ambiguous_depts(department, index)
            if ambiguous:
                lines = [f"'{department}' is ambiguous. Please specify:"]
                for i, (en, zh, count) in enumerate(ambiguous[:5], 1):
                    lines.append(f"  {i}. {en} ({zh})")
                return "\n".join(lines)
            return f"Department '{department}' not found."
        courses = _collect_courses(_dept_docs(index, dept_zh))
        if not courses:
            return f"No credit-bearing courses found for {display}."
        return _build_course_plan(courses, target_credits, display)

    @tool
    def plan_eecs_courses_by_credits(department: str, target_credits: int) -> str:
        """
        Select EECS courses (CS, EE, CE, NLT, AI) whose credits sum to target_credits.

        Args:
            department:     Full department name in English or Chinese.
            target_credits: Desired total credits e.g. 24.
        """
        if err := _check_index(index): return err
        dept_zh, display = _resolve_dept(department)
        if dept_zh is None:
            return f"Department '{department}' not found."
        if dept_zh not in EECS_DEPTS_ZH:
            return f"'{display}' is outside EECS scope. I only cover CS, EE, CE, NLT, and AI."
        courses = _collect_courses(_dept_docs(index, dept_zh))
        if not courses:
            return f"No credit-bearing courses found for {display}."
        return _build_course_plan(courses, target_credits, display)

    # ── Graph / multi-filter search ───────────────────────────────────────────

    def _graph_search(
        docs:        list[dict],
        scope_label: str,
        department:  Optional[str],
        building:    Optional[str],
        weekday:     Optional[str],
        period:      Optional[str],
        credits:     Optional[int],
        instructor:  Optional[str],
        course_name: Optional[str],
        req_type:    Optional[str],
    ) -> str:
        parts = []
        if department:  parts.append(f"dept={department}")
        if building:    parts.append(f"building={building}")
        if weekday:     parts.append(f"day={weekday}")
        if period:      parts.append(f"period={period}")
        if credits:     parts.append(f"credits={credits}")
        if instructor:  parts.append(f"instructor={instructor}")
        if course_name: parts.append(f"course={course_name}")
        if req_type:    parts.append(f"type={req_type}")
        if not parts:
            return "Please provide at least one filter."

        query_desc = " + ".join(parts)
        filtered   = _apply_filters(
            docs=docs, department=department, building=building,
            weekday=weekday, period=period, credits=credits,
            instructor=instructor, course_name=course_name, req_type=req_type,
        )

        if not filtered and course_name:
            sem_results = index.search(course_name, top_k=10)
            doc_lookup  = {d.get("text"): d for d in docs}
            sem_docs    = [doc_lookup[r.text] for r in sem_results if r.text in doc_lookup]
            filtered    = _apply_filters(
                docs=sem_docs, department=department, building=building,
                weekday=weekday, period=period, credits=credits,
                instructor=instructor, req_type=req_type,
            ) or sem_docs

        return _format_graph_results(filtered, query_desc)

    @tool
    def graph_search_courses(
        department:  Optional[str] = None,
        building:    Optional[str] = None,
        weekday:     Optional[str] = None,
        period:      Optional[str] = None,
        credits:     Optional[int] = None,
        instructor:  Optional[str] = None,
        course_name: Optional[str] = None,
        req_type:    Optional[str] = None,
    ) -> str:
        """
        Multi-hop graph search combining any filters as AND conditions across all NCU courses.

        Args:
            department:  Full dept name e.g. "computer science"
            building:    Building code/name e.g. "E6"
            weekday:     Day e.g. "Monday", "thursday"
            period:      Period e.g. "3", "A"
            credits:     Exact credit count e.g. 3
            instructor:  Instructor name (partial)
            course_name: Course name or code keyword (partial)
            req_type:    "required" or "elective"
        """
        if err := _check_index(index): return err
        if department:
            dept_zh, _ = _resolve_dept(department)
            if dept_zh is None:
                ambiguous = _find_ambiguous_depts(department, index)
                if len(ambiguous) > 1:
                    lines = [f"'{department}' matches multiple departments:"]
                    for i, (en, zh, count) in enumerate(ambiguous, 1):
                        lines.append(f"  {i}. {en} ({zh}) — {count} courses")
                    lines.append("Please specify which department you mean.")
                    return "\n".join(lines)
        if course_name:
            course_matches = _find_similar_courses(course_name, index)
            if len(course_matches) > 5:
                lines = [f"'{course_name}' matches {len(course_matches)} courses:"]
                for i, (name, code, dept) in enumerate(course_matches[:5], 1):
                    lines.append(f"  {i}. [{code}] {name} — {DEPT_EN_LABEL.get(dept, dept)}")
                lines.append(f"  … and {len(course_matches)-5} more. Please be more specific.")
                return "\n".join(lines)
        return _graph_search(
            docs=index.documents, scope_label="all NCU",
            department=department, building=building, weekday=weekday,
            period=period, credits=credits, instructor=instructor,
            course_name=course_name, req_type=req_type,
        )

    @tool
    def graph_search_eecs_courses(
        department:  Optional[str] = None,
        building:    Optional[str] = None,
        weekday:     Optional[str] = None,
        period:      Optional[str] = None,
        credits:     Optional[int] = None,
        instructor:  Optional[str] = None,
        course_name: Optional[str] = None,
        req_type:    Optional[str] = None,
    ) -> str:
        """
        Multi-hop graph search across EECS courses (CS, EE, CE, NLT, AI) only.

        Args:
            department:  Full dept name e.g. "computer science"
            building:    Building code/name e.g. "E6"
            weekday:     Day e.g. "Monday", "thursday"
            period:      Period e.g. "3", "A"
            credits:     Exact credit count e.g. 3
            instructor:  Instructor name (partial)
            course_name: Course name or code keyword (partial)
            req_type:    "required" or "elective"
        """
        if err := _check_index(index): return err
        if department:
            dept_zh, display = _resolve_dept(department)
            if dept_zh and dept_zh not in EECS_DEPTS_ZH:
                return f"'{display}' is outside EECS scope. I only cover CS, EE, CE, NLT, and AI."
        return _graph_search(
            docs=_eecs_docs(index), scope_label="EECS",
            department=department, building=building, weekday=weekday,
            period=period, credits=credits, instructor=instructor,
            course_name=course_name, req_type=req_type,
        )

    # ── Listing helpers ───────────────────────────────────────────────────────

    @tool
    def list_departments() -> str:
        """List all NCU departments in the index with course counts."""
        if err := _check_index(index): return err
        counts: dict[str, int] = {}
        for d in index.documents:
            zh = d.get("dept_name_zh", "")
            if zh:
                counts[zh] = counts.get(zh, 0) + 1
        lines = [f"{DEPT_EN_LABEL.get(zh, zh)} ({zh}) [{n}]" for zh, n in counts.items()]
        lines.sort()
        return "NCU departments:\n" + "\n".join(lines)

    @tool
    def list_eecs_departments() -> str:
        """List all EECS departments in the index with course counts."""
        if err := _check_index(index): return err
        counts: dict[str, int] = {}
        for d in _eecs_docs(index):
            zh = d.get("dept_name_zh", "")
            if zh:
                counts[zh] = counts.get(zh, 0) + 1
        lines = [f"{DEPT_EN_LABEL.get(zh, zh)} ({zh}) [{n}]" for zh, n in counts.items()]
        lines.sort()
        return "EECS departments:\n" + "\n".join(lines)

    @tool
    def search_courses_by_time(day: str, period: Optional[str] = None) -> str:
        """
        Find courses on a weekday, optionally filtered by period.

        Args:
            day:    Weekday e.g. 'Monday', 'Friday'.
            period: Optional period code e.g. '3', 'A'.
        """
        if err := _check_index(index): return err
        day_clean = day.strip().lower()
        if day_clean not in VALID_DAYS:
            return f"Invalid day '{day}'. Use: {', '.join(sorted(VALID_DAYS))}"
        period_clean = period.strip().upper() if period else None
        candidates   = [
            d for d in index.documents
            if day_clean in [w.lower() for w in d.get("weekdays", [])]
        ]
        if period_clean:
            candidates = [
                d for d in candidates
                if period_clean in [p.upper() for p in d.get("periods", [])]
            ]
        label = f"{day.title()} P{period_clean}" if period_clean else day.title()
        return _format_results([d["text"] for d in candidates], label)

    @tool
    def search_courses_by_location(building: str) -> str:
        """
        Find courses in a building or room.

        Args:
            building: Building name or code e.g. 'E6', 'Language Center'.
        """
        if err := _check_index(index): return err
        bld = building.strip().lower()
        results = [
            d["text"] for d in index.documents
            if any(bld in b.lower() for b in d.get("buildings", []))
            or any(bld in c.lower() for c in d.get("classrooms", []))
        ]
        return _format_results(results, building)

    @tool
    def list_available_days() -> str:
        """List all weekdays that have scheduled courses."""
        if err := _check_index(index): return err
        WEEKDAY_ZH_TO_EN = {
            "星期一": "Monday",    "monday": "Monday",
            "星期二": "Tuesday",   "tuesday": "Tuesday",
            "星期三": "Wednesday", "wednesday": "Wednesday",
            "星期四": "Thursday",  "thursday": "Thursday",
            "星期五": "Friday",    "friday": "Friday",
            "星期六": "Saturday",  "saturday": "Saturday",
            "星期日": "Sunday",    "sunday": "Sunday",
        }
        days: set[str] = set()
        for d in index.documents:
            for w in d.get("weekdays", []):
                if not w:
                    continue
                days.add(WEEKDAY_ZH_TO_EN.get(w.lower(), WEEKDAY_ZH_TO_EN.get(w, w.title())))
        return "Days with courses: " + ", ".join(sorted(days)) if days else "No schedule data."

    return [
        clarify,
        detect_ambiguity,
        detect_ambiguity_eecs,
        search_courses_by_content,
        search_eecs_courses_by_content,
        search_courses_by_department,
        get_all_courses_by_department,
        get_all_eecs_courses_by_department,
        plan_courses_by_credits,
        plan_eecs_courses_by_credits,
        graph_search_courses,
        graph_search_eecs_courses,
        list_departments,
        list_eecs_departments,
        search_courses_by_time,
        search_courses_by_location,
        list_available_days,
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from core import LocalEmbedder, VectorIndex

    embedder = LocalEmbedder()
    idx      = VectorIndex.load("ncu_index.pkl", embedder)
    tools    = create_tools(idx)
    tmap     = {t.name: t for t in tools}
    print(f"Loaded {len(idx)} docs | {len(tools)} tools\n")

    cases = [
        ("detect_ambiguity",                    {"query": "communication"}),
        ("detect_ambiguity_eecs",               {"query": "computer science"}),
        ("list_eecs_departments",               {}),
        ("get_all_eecs_courses_by_department",  {"department": "communication engineering"}),
        ("plan_eecs_courses_by_credits",        {"department": "network learning technology", "target_credits": 9}),
        ("graph_search_eecs_courses",           {"course_name": "Engineering Mathematics"}),
        ("graph_search_courses",                {"department": "computer science", "credits": 3}),
    ]
    for name, args in cases:
        print(f"=== {name}({args}) ===")
        print(tmap[name].invoke(args))
        print()