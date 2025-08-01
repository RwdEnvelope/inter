# graph/interview_flow_live.py
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import TypedDict, List, Optional

# 项目根路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ==== 业务逻辑 ====
from agents.question_agent import question_agent
from tools.analysis   import start_av_recording, stop_av_recording
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage

# ==== LangGraph ====
from langgraph.graph import StateGraph, START, END

# ------------------ 状态 ------------------
class InterviewState(TypedDict):
    resume: str
    round: int
    messages: List[AnyMessage]
    should_continue: bool

    # 累积数据
    qa_pairs:       List[tuple[str, str]]
    audio_summaries: List[str]
    video_summaries: List[str]

# ------------------ 节点 ------------------
def ask(state: InterviewState) -> InterviewState:
    """面试官提问"""
    return question_agent(state)

def start_rec(state: InterviewState) -> InterviewState:
    """一旦提问就开始录制"""
    print("▶️ 正在录制音视频…（回答完按 Enter 停止）")
    start_av_recording()
    return {}  # 无需写回

def listen_and_stop(state: InterviewState) -> InterviewState:
    """展示问题→收答案→停止录制并分析"""
    # 1. 打印面试官问题
    ai_msg = next((m for m in state["messages"] if isinstance(m, AIMessage)), None)
    question_text = ai_msg.content if ai_msg else "(问题丢失)"
    print(f"\n面试官：{question_text}")

    # 2. 候选人回答
    answer_text = input("候选人：").strip()

    # 3. 结束录制并拿分析结果
    result = stop_av_recording()
    print("✅ 录制/分析完成\n")

    # 4. 返回增量：HumanMessage + QA + 摘要
    return {
        "messages": [HumanMessage(content=answer_text)],
        "qa_pairs": state.get("qa_pairs", []) + [(question_text, answer_text)],
        "audio_summaries": state.get("audio_summaries", []) + [result.get("audio_summary", "")],
        "video_summaries": state.get("video_summaries", []) + [result.get("video_summary", "")],
    }

def need_more(state: InterviewState) -> bool:
    """是否继续？"""
    return state.get("should_continue", False) and state.get("round", 0) < 10

# ------------------ 构建 LangGraph ------------------
g = StateGraph(InterviewState)
g.add_node("ask",      ask)
g.add_node("startrec", start_rec)
g.add_node("listen",   listen_and_stop)

g.add_edge(START, "ask")
g.add_conditional_edges("ask", need_more, {True: "startrec", False: END})
g.add_edge("startrec", "listen")
g.add_edge("listen",   "ask")

interview_graph = g.compile()

# ------------------ CLI 演示 ------------------
if __name__ == "__main__":
    sample_resume = """姓名：Alice
学历：计算机科学本科
经历：阿里巴巴推荐系统实习 6 个月
技能：Python / PyTorch"""

    init_state: InterviewState = {
        "resume": sample_resume,
        "round": 0,
        "messages": [],
        "should_continue": True,
        "qa_pairs": [],
        "audio_summaries": [],
        "video_summaries": [],
    }

    final = interview_graph.invoke(init_state)

    # ===== 整场结果展示 =====
    print("\n=== 面试结束，完整问答回顾 ===")
    for i, (q, a) in enumerate(final["qa_pairs"], 1):
        print(f"\n【第{i}问】{q}\n【答】{a}")

    print("\n📝 全场音频摘要：")
    print("\n".join(filter(None, final.get("audio_summaries", []))) or "无")

    print("\n🎞️ 全场视频摘要：")
    print("\n".join(filter(None, final.get("video_summaries", []))) or "无")
