import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ---------- 状态 ----------
class AgentState(TypedDict):
    resume: str
    round: int
    messages: Annotated[List[AnyMessage], add_messages]
    should_continue: bool

# ---------- Agent ----------
def question_agent(state: AgentState) -> AgentState:
    resume  = state["resume"]
    round_i = state.get("round", 0)
    history = state["messages"]

    # -------- 首轮 --------
    if round_i == 0:
        return {
            **state,
            "round": 1,
            "should_continue": True,
            "messages": [AIMessage(content="请先自我介绍一下。")],
        }

    # -------- 抽取上一轮回答 --------
    last_answer = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )

    prompt = (
        "你是一名资深中文面试官。\n\n"
        f"已知：\n- 候选人简历：\n{resume}\n"
        f"- 候选人上一轮回答：\n{last_answer}\n\n"
        "任务：\n"
        "1. 如果上一轮回答有可深挖之处，基于该回答追问 1 个问题；\n"
        "2. 否则，从简历中选一处尚未被问到的要点，提出 1 个全新问题；\n"
        "3. 若已没有可问的问题，仅输出“结束”。\n"
        "⚠️ 只输出一句完整提问，不要编号，不要额外说明。"
    )

    reply = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是专业中文面试官。"},
            {"role": "user",    "content": prompt},
        ],
    ).choices[0].message.content.strip()

    # -------- 判断结束 --------
    if reply == "结束" or "没有更多问题" in reply:
        return {
            **state,
            "should_continue": False,
            "messages": [AIMessage(content="谢谢你的回答，我这边没有其他问题了。")],
        }

    # -------- 正常追问 --------
    return {
        **state,
        "round": round_i + 1,
        "should_continue": True,
        "messages": [AIMessage(content=reply)],
    }
