# ncu_rag_agent.py — Plan-then-Execute pattern

from __future__ import annotations
from langgraph.checkpoint.memory import MemorySaver
import re
import os
import json
import time
import logging
import operator
from typing import Annotated, List, TypedDict, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage,
)
from langgraph.graph import StateGraph, END

from core import LocalEmbedder, VectorIndex
from tools import create_tools, EECS_DEPTS_ZH   # single source of truth

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

if PROVIDER == "ollama":
    MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    API_KEY  = "ollama"
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
elif PROVIDER == "gemini":
    MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    API_KEY  = os.getenv("GEMINI_API_KEY", "")
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
elif PROVIDER == "groq":
    MODEL    = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    API_KEY  = os.getenv("GROQ_API_KEY", "")
    BASE_URL = "https://api.groq.com/openai/v1"
else:
    raise ValueError(f"Unsupported PROVIDER: {PROVIDER}")

INDEX_FILE          = "ncu_index.pkl"
MAX_REQUEST_RETRIES = 3
RETRY_WAIT_429      = 60
RETRY_WAIT_OTHER    = 3
THINK_TAG_RE        = re.compile(r"<think>.*?</think>", re.DOTALL)

# Max non-system messages kept in conversation history (10 exchanges = 20 msgs)
MAX_HISTORY_MESSAGES = 20

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the NCU EECS Course Assistant.

Scope: ONLY answer about these departments:
  1. Computer Science and Engineering (資訊工程學系)
  2. Electrical Engineering (電機工程學系)
  3. Communication Engineering (通訊工程學系)
  4. College of EECS Bachelor's Program (資訊電機學院學士班)
  5. Graduate Institute of Network Learning Technology (網路學習科技研究所碩士班)
  6. International Master's in Artificial Intelligence (人工智慧國際碩士學位學程)

Available tools (prefer the *_eecs_* variants for EECS-specific queries):
  Ambiguity:
    detect_ambiguity(query)              — all NCU departments
    detect_ambiguity_eecs(query)         — EECS departments only
  Content search:
    search_courses_by_content(query)     — all NCU courses
    search_eecs_courses_by_content(query)— EECS courses only
  Department search:
    search_courses_by_department(department, keyword)
  Full listings:
    get_all_courses_by_department(department)
    get_all_eecs_courses_by_department(department)
  Credit planning:
    plan_courses_by_credits(department, target_credits)
    plan_eecs_courses_by_credits(department, target_credits)
  Multi-filter search:
    graph_search_courses(department, building, weekday, period,
                         credits, instructor, course_name, req_type)
    graph_search_eecs_courses(department, building, weekday, period,
                              credits, instructor, course_name, req_type)
  Listing helpers:
    list_departments()
    list_eecs_departments()
    search_courses_by_time(day, period)
    search_courses_by_location(building)
    list_available_days()
  Clarification:
    clarify(question)

Use full department names: "computer science and engineering", "electrical engineering".
Respond in the same language as the user.
"""

PLAN_PROMPT = """\
You are a query planner for an NCU EECS course assistant.

Given a user question, produce an execution plan as a JSON array.
Each step has: {{"tool": "<tool_name>", "args": {{...}}, "reason": "<why>"}}

Rules:
- Prefer *_eecs_* tool variants for EECS-specific queries.
- Use detect_ambiguity_eecs FIRST if the query could match multiple EECS departments or courses.
- Use clarify if disambiguation finds multiple matches.
- For credit planning: use plan_eecs_courses_by_credits.
- For full dept listing: use get_all_eecs_courses_by_department.
- For multi-dimensional EECS queries: use graph_search_eecs_courses.
- For simple topic search: use search_eecs_courses_by_content.
- Maximum 3 steps. Most queries need only 1.
- Never include a synthesise step — that is done separately.
- Omit optional args that are not needed (do not emit null values).

Available tools and signatures:
  detect_ambiguity_eecs(query: str)
  detect_ambiguity(query: str)
  clarify(question: str)
  search_eecs_courses_by_content(query: str)
  search_courses_by_content(query: str)
  search_courses_by_department(department: str, keyword: str)
  get_all_eecs_courses_by_department(department: str)
  get_all_courses_by_department(department: str)
  plan_eecs_courses_by_credits(department: str, target_credits: int)
  plan_courses_by_credits(department: str, target_credits: int)
  graph_search_eecs_courses(department: str, building: str, weekday: str,
                             period: str, credits: int, instructor: str,
                             course_name: str, req_type: str)
  graph_search_courses(department: str, building: str, weekday: str,
                       period: str, credits: int, instructor: str,
                       course_name: str, req_type: str)
  list_eecs_departments()
  list_departments()
  search_courses_by_time(day: str, period: str)
  search_courses_by_location(building: str)
  list_available_days()

Output ONLY a valid JSON array. No explanation, no markdown fences.

Examples:
  Q: "CS courses on Monday"
  A: [{{"tool":"graph_search_eecs_courses","args":{{"department":"computer science and engineering","weekday":"Monday"}},"reason":"multi-dim EECS search by dept+day"}}]

  Q: "communication courses"
  A: [{{"tool":"detect_ambiguity_eecs","args":{{"query":"communication"}},"reason":"ambiguous within EECS: could be comm engineering or language center"}}]

  Q: "24 credits from EE"
  A: [{{"tool":"plan_eecs_courses_by_credits","args":{{"department":"electrical engineering","target_credits":24}},"reason":"EECS credit planning"}}]

  Q: "where is Engineering Mathematics"
  A: [{{"tool":"graph_search_eecs_courses","args":{{"course_name":"Engineering Mathematics"}},"reason":"find EECS course location"}}]

User question: {question}
"""

SYNTHESISE_PROMPT = """\
You are the NCU EECS Course Assistant. Answer the user's question using ONLY the tool results below.

Rules:
- Use ONLY data from tool results. Never guess or invent.
- List courses as: [CODE] Name — N credits (Type)
- If no data found, say so clearly.
- If clarification was needed, present the options to the user.
- Respond in the same language as the user's question.

User question: {question}

Tool results:
{tool_results}

Answer:"""

SCOPE_INTENT_PROMPT = """\
Classify the user message for an NCU EECS course assistant.
EECS covers: Computer Science, Electrical Engineering, Communication Engineering,
Network Learning Technology, Artificial Intelligence.

Reply with EXACTLY one word:
  COURSE       → asks about EECS courses, credits, schedule, professors, departments, buildings
  OUT_OF_SCOPE → asks about a department clearly outside EECS (math, physics, etc.)
  CHITCHAT     → greeting, thanks, small talk
  OFF_TOPIC    → completely unrelated to NCU courses

User message: {message}
"""

CHITCHAT_REPLIES = {
    "CHITCHAT": (
        "Hi! I'm the NCU EECS Course Assistant. "
        "Ask me about CS, EE, or Communication Engineering courses."
    ),
    "OFF_TOPIC": (
        "Sorry, I only cover NCU EECS — CS, EE, CE, NLT, and AI programs."
    ),
    "OUT_OF_SCOPE": (
        "That department is outside my scope. I only cover NCU EECS:\n"
        "• Computer Science and Engineering\n"
        "• Electrical Engineering\n"
        "• Communication Engineering\n"
        "• Network Learning Technology\n"
        "• International Master's in AI\n\n"
        "For other departments: https://cis.ncu.edu.tw"
    ),
}

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    plan:     Optional[list]    # serialised step dicts
    results:  Optional[list]    # serialised result dicts
    question: str

# ── Index ─────────────────────────────────────────────────────────────────────

def load_index() -> VectorIndex | None:
    embedder = LocalEmbedder()
    if not Path(INDEX_FILE).exists():
        logger.warning(f"'{INDEX_FILE}' not found. Run: python index_builder.py")
        return None
    try:
        idx = VectorIndex.load(INDEX_FILE, embedder)
        logger.info(f"Index: {idx}")
        return idx
    except Exception as exc:
        logger.error(f"Index load failed: {exc}")
        return None

# ── Model factory ─────────────────────────────────────────────────────────────

def make_base_model(temperature: float = 0) -> ChatOpenAI:
    if PROVIDER != "ollama" and not API_KEY:
        raise ValueError(f"API_KEY for {PROVIDER.upper()} not set in .env")
    return ChatOpenAI(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        temperature=temperature, max_retries=0,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    return THINK_TAG_RE.sub("", text).strip()


def classify_intent(message: str, model: ChatOpenAI) -> str:
    try:
        resp   = model.invoke([HumanMessage(
            content=SCOPE_INTENT_PROMPT.format(message=message)
        )])
        intent = clean(resp.content).strip().upper().split()[0]
        return intent if intent in ("COURSE", "OUT_OF_SCOPE", "CHITCHAT", "OFF_TOPIC") \
               else "COURSE"
    except Exception as exc:
        logger.warning(f"Intent classification failed: {exc}")
        return "COURSE"


def extract_failed_generation(err_str: str) -> str | None:
    for pattern in [
        r"'failed_generation':\s*'(.*?)'(?:\s*[,}])",
        r'"failed_generation":\s*"(.*?)"(?:\s*[,}])',
    ]:
        m = re.search(pattern, err_str, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _handle_error(err: str) -> tuple[bool, int, str | None]:
    if "429" in err:
        print(f"\n\033[33m[Rate limit]\033[0m Waiting {RETRY_WAIT_429}s…")
        return True, RETRY_WAIT_429, None
    if "400" in err and "tool_use_failed" in err:
        rescued = extract_failed_generation(err)
        if rescued:
            return False, 0, rescued
        return True, RETRY_WAIT_OTHER, None
    if any(c in err for c in ["500", "502", "503", "Connection", "Timeout"]):
        return True, RETRY_WAIT_OTHER, None
    return False, 0, None

# ── Plan-then-Execute nodes ───────────────────────────────────────────────────

def make_planner_node(planner_model: ChatOpenAI):
    """Node 1: produce a JSON execution plan from the user question."""
    def plan(state: AgentState) -> dict:
        question = state["question"]
        logger.info(f"  [Plan] Generating plan for: {question!r}")

        prompt = PLAN_PROMPT.format(question=question)
        try:
            resp = planner_model.invoke([
                SystemMessage(content="You are a query planner. Output only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = clean(resp.content).strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            steps_data = json.loads(raw)
        except Exception as exc:
            logger.warning(f"Plan parsing failed: {exc} — using default search")
            steps_data = [{
                "tool":   "search_eecs_courses_by_content",
                "args":   {"query": question},
                "reason": "fallback",
            }]

        steps = []
        for s in steps_data[:3]:
            tool_name = s.get("tool", "")
            # Drop null/None args so tool signatures stay clean
            args      = {k: v for k, v in s.get("args", {}).items() if v is not None}
            reason    = s.get("reason", "")
            steps.append({"tool": tool_name, "args": args, "reason": reason})
            logger.info(f"  [Plan] Step: {tool_name}({args}) — {reason}")

        # Always emit a non-empty plan so the executor always runs
        return {"plan": steps or [{"tool": "search_eecs_courses_by_content",
                                   "args": {"query": question},
                                   "reason": "empty-plan fallback"}],
                "results": []}

    return plan


def make_executor_node(tool_executor: dict):
    """Node 2: execute each planned step deterministically."""
    def execute(state: AgentState) -> dict:
        plan    = state.get("plan", [])
        results = list(state.get("results") or [])

        for step in plan:
            tool_name = step["tool"]
            args      = step["args"]
            logger.info(f"  → Execute: {tool_name}({args})")
            print(
                f"  \033[33m[Action]\033[0m  "
                f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
            )

            try:
                fn = tool_executor.get(tool_name)
                if fn is None:
                    obs, success = f"Tool '{tool_name}' not found.", False
                else:
                    obs, success = fn.invoke(args), True
            except Exception as exc:
                obs, success = f"Tool error: {exc}", False
                logger.exception(f"Tool failed: {tool_name}")

            obs_str = str(obs)
            snippet = obs_str[:120].replace("\n", " ")
            dots    = "…" if len(obs_str) > 120 else ""
            print(f"  \033[36m[Result]\033[0m  {snippet}{dots}")

            results.append({
                "tool":    tool_name,
                "args":    args,
                "output":  obs_str,
                "success": success,
            })

        return {"results": results}

    return execute


def make_synthesiser_node(synth_model: ChatOpenAI):
    """Node 3: synthesise a final answer from all tool results."""
    def synthesise(state: AgentState) -> dict:
        question = state["question"]
        results  = state.get("results") or []

        if not results:
            answer = "I couldn't find any information. Please try rephrasing your question."
        else:
            tool_results_text = "\n\n".join(
                f"[{r['tool']}({r['args']})]\n{r['output']}"
                for r in results
            )
            prompt = SYNTHESISE_PROMPT.format(
                question=question,
                tool_results=tool_results_text,
            )
            try:
                resp   = synth_model.invoke([HumanMessage(content=prompt)])
                answer = clean(resp.content).strip()
            except Exception as exc:
                logger.warning(f"Synthesis failed: {exc}")
                answer = "\n\n".join(r["output"] for r in results if r["success"])

        logger.info(f"  [Synthesise] Answer length: {len(answer)} chars")
        return {"messages": [AIMessage(content=answer)]}

    return synthesise

# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(tools: list):
    base_model = make_base_model(temperature=0)
    executor_tools = {t.name: t for t in tools}

    wf = StateGraph(AgentState)
    wf.add_node("plan", make_planner_node(base_model))
    wf.add_node("execute", make_executor_node(executor_tools))
    wf.add_node("synthesise", make_synthesiser_node(base_model))

    wf.set_entry_point("plan")
    wf.add_edge("plan", "execute")
    wf.add_edge("execute", "synthesise")
    wf.add_edge("synthesise", END)

    memory = MemorySaver()
    # return wf.compile(checkpointer=memory)
    return wf.compile()

# ── REPL ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if PROVIDER != "ollama" and not API_KEY:
        print(f"\033[31m[Error]\033[0m {PROVIDER.upper()}_API_KEY not set in .env")
        return

    index        = load_index()
    tools        = create_tools(index)
    app          = build_graph(tools)
    intent_model = make_base_model()

    print("\n\033[1m━━━  NCU EECS Course Assistant  ━━━\033[0m")
    print(f"  Provider : {PROVIDER}")
    print(f"  Model    : {MODEL}")
    print(f"  Pattern  : Plan-then-Execute")
    print(f"  Index    : {'✓ ' + str(len(index)) + ' docs' if index else '✗ not found'}")
    print("  Type 'exit' or 'quit' to leave.\n")

    conversation: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # ── Intent gate ───────────────────────────────────────────────────────
        intent = classify_intent(user_input, intent_model)
        logger.info(f"Intent: {intent}")
        if intent != "COURSE":
            reply = CHITCHAT_REPLIES.get(intent, CHITCHAT_REPLIES["OFF_TOPIC"])
            print(f"\n\033[32mAssistant:\033[0m {reply}\n")
            continue

        conversation.append(HumanMessage(content=user_input))

        # Fresh state per turn — no cross-turn state pollution
        state: AgentState = {
            "messages": list(conversation),
            "plan":     None,
            "results":  None,
            "question": user_input,
        }

        answered = False
        for attempt in range(1, MAX_REQUEST_RETRIES + 1):
            try:
                final = app.invoke(
                    state,
                    configurable={
                        "thread_id": "default-thread",
                        "checkpoint_ns": "ncu-eecs",
                        "checkpoint_id": f"turn-{attempt}"
                    }
                )
                answered = True

                ai_msgs = [
                    m for m in final.get("messages", [])
                    if isinstance(m, AIMessage) and m.content.strip()
                ]
                if ai_msgs:
                    answer = clean(ai_msgs[-1].content)
                    print(f"\n\033[32mAssistant:\033[0m {answer}\n")
                    conversation.append(AIMessage(content=answer))
                else:
                    results = final.get("results") or []
                    raw = "\n\n".join(r["output"] for r in results if r.get("success"))
                    if raw:
                        print(f"\n\033[32mAssistant:\033[0m {raw[:600]}\n")
                break

            except Exception as exc:
                err = str(exc)
                logger.error(f"Error (attempt {attempt}): {err}")
                should_retry, wait, rescued = _handle_error(err)

                if rescued:
                    answer = clean(rescued)
                    print(f"\n\033[32mAssistant:\033[0m {answer}\n")
                    conversation.append(AIMessage(content=answer))
                    answered = True
                    break

                if not should_retry or attempt == MAX_REQUEST_RETRIES:
                    print(f"\n\033[31m[Error]\033[0m {err}\n")
                    # Roll back the human message we optimistically appended
                    conversation = [
                        m for m in conversation
                        if not (isinstance(m, HumanMessage) and m.content == user_input)
                    ]
                    break

                time.sleep(wait)
                print(f"\033[33m[Retry {attempt}/{MAX_REQUEST_RETRIES}]\033[0m Retrying…")

        if not answered:
            continue

        # Keep history to last MAX_HISTORY_MESSAGES non-system messages
        system_msgs = [m for m in conversation if isinstance(m, SystemMessage)]
        other_msgs  = [m for m in conversation if not isinstance(m, SystemMessage)]
        conversation = system_msgs + other_msgs[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()