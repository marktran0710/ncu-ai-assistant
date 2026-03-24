# test_model_tool_calling.py
# Kiểm tra model có follow tool output không

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:32b") if PROVIDER == "ollama" \
           else os.getenv("GROQ_MODEL")
API_KEY  = "ollama" if PROVIDER == "ollama" else os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") \
           if PROVIDER == "ollama" else "https://api.groq.com/openai/v1"

model = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)

FAKE_TOOL_OUTPUT = """Found 2 result(s) for 'Introduction to Computer Science II':

[1] Course: Introduction to Computer Science II [CE1002]
Department: 資訊工程學系
Instructor: 施國琛
Credits: 3
Type: Required
Schedule: Thursday (13:00-13:50, 14:00-14:50, 15:00-15:50)
Classroom: E6-A207 (Engineering Building 7)

[2] Course: Introduction to Computer Science II Lab [CE1004-A]
Department: 資訊工程學系
Instructor: 施國琛
Credits: 1
Type: Required
Schedule: Friday (15:00-15:50, 16:00-16:50)
Classroom: E6-A206 (Engineering Building 7)"""

SYSTEM = """You are an NCU course assistant.
When tool output is provided, answer DIRECTLY from it.
Do NOT suggest calling more tools. Do NOT say you lack data.
"""

print(f"Testing model: {MODEL}")
print("=" * 55)

TESTS = [
    {
        "name": "Where is Introduction to Computer Science II?",
        "question": "Where is Introduction to Computer Science II?",
        "check": ["E6-A207", "Engineering Building 7", "E6-A206"],
    },
    {
        "name": "How many credits?",
        "question": "How many credits does Introduction to Computer Science II have?",
        "check": ["3"],
    },
    {
        "name": "Who teaches it?",
        "question": "Who teaches Introduction to Computer Science II?",
        "check": ["施國琛"],
    },
    {
        "name": "What day is it?",
        "question": "What day is Introduction to Computer Science II on?",
        "check": ["Thursday", "Friday"],
    },
]

passed = 0
for test in TESTS:
    # Simulate: HumanMessage → AIMessage with tool_call → ToolMessage → model synthesises
    fake_ai = AIMessage(
        content="",
        tool_calls=[{
            "id":   "tc_test_001",
            "name": "search_courses_by_content",
            "args": {"query": test["question"]},
        }]
    )
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=test["question"]),
        fake_ai,
        ToolMessage(tool_call_id="tc_test_001", content=FAKE_TOOL_OUTPUT),
    ]
    response = model.invoke(msgs)
    answer   = response.content.strip()

    ok = any(kw in answer for kw in test["check"])
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1

    print(f"\n[{status}] {test['name']}")
    print(f"  Expected one of: {test['check']}")
    print(f"  Got: {answer[:150]}")

print(f"\n{'='*55}")
print(f"Result: {passed}/{len(TESTS)} passed")

if passed == len(TESTS):
    print("Model handles tool output correctly — agent should work well.")
elif passed >= 2:
    print("Model partially follows tool output — may still hallucinate sometimes.")
else:
    print("Model ignores tool output — try a different/larger model.")