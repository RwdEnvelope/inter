import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from tools.analysis import start_av_recording_tool, stop_av_recording_tool
from agents.question_agent import question_agent  # 已支持工具
from langgraph.prebuilt import ToolNode

# 1. 定义状态结构
class AgentState(TypedDict):
    resume: str
    messages: list[BaseMessage]
    round: int
    audio_summary: Optional[str]
    video_summary: Optional[str]
    audio_dir: Optional[str]
    video_dir: Optional[str]
    should_continue: bool

# 2. 绑定工具节点
start_tool_node = ToolNode([start_av_recording_tool])
stop_tool_node = ToolNode([stop_av_recording_tool])

# 3. 模拟前端按钮等待（实际可用轮询事件）
def wait_for_user_click(state: AgentState):
    input("🟡 等待用户答完题，按 Enter 停止录制...")
    return {}

# 4. 处理工具返回 observation（来自 stop_av_recording_tool）
def extract_observation(state: AgentState, tool_output: dict) -> AgentState:
    return {
        **state,
        "audio_summary": tool_output.get("audio_summary", ""),
        "video_summary": tool_output.get("video_summary", ""),
        "audio_dir": tool_output.get("audio_dir"),
        "video_dir": tool_output.get("video_dir"),
    }

# 5. 提问节点（加上分析观察）
def question_agent_node(state: AgentState) -> AgentState:
    observation = (
        f"[音频分析]：{state.get('audio_summary', '')}\n"
        f"[视频分析]：{state.get('video_summary', '')}"
        if state.get("audio_summary") else ""
    )

    messages = state["messages"]
    if observation:
        messages.append(HumanMessage(content=observation))

    reply = question_agent.invoke(state)

    return {
        **state,
        "messages": messages + [reply["messages"][-1]],
        "round": state["round"] + 1,
        "should_continue": reply["should_continue"] and state["round"] < 9
    }

# 6. 条件边：是否继续
def route_decision(state: AgentState) -> str:
    return "continue" if state["should_continue"] else "end"

# 7. 构建 LangGraph 流程
builder = StateGraph(AgentState)

builder.add_node("question", question_agent_node)
builder.add_node("start_tool", start_tool_node)
builder.add_node("wait_for_click", wait_for_user_click)
builder.add_node("stop_tool", stop_tool_node)
builder.add_node("update_observation", extract_observation)

builder.set_entry_point("question")
builder.add_edge("question", "start_tool")
builder.add_edge("start_tool", "wait_for_click")
builder.add_edge("wait_for_click", "stop_tool")
builder.add_edge("stop_tool", "update_observation")

builder.add_conditional_edges(
    "update_observation",
    route_decision,
    {
        "continue": "question",
        "end": END
    }
)

graph = builder.compile()

# 8. 启动测试用初始状态（第 0 轮自我介绍）
if __name__ == "__main__":
    init_state: AgentState = {
        "resume": "姓名：张三\n学历：计算机科学与技术本科\n经历：在腾讯实习过，做过大模型微调项目。",
        "messages": [HumanMessage(content="请做一次自我介绍")],
        "round": 0,
        "audio_summary": "",
        "video_summary": "",
        "audio_dir": None,
        "video_dir": None,
        "should_continue": True
    }

    final_state = graph.invoke(init_state)
    print("✅ 面试流程已结束")
