from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

# ◆ 把项目根目录加入 sys.path，保证 `tools.*` 可以被正确 import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import Tool

# ----------------------------------------------------------------------
# ① 复用你已有的启动 / 停止函数
#    如果 tools.analysis 路径不同，请相应修改 import
from tools.analysis import start_av_recording, stop_av_recording
# ----------------------------------------------------------------------

# ② 定义子图状态（可按需扩展）
class AVState(dict):
    """graph.invoke 时传入/产出的状态 dict"""
    recording_started: bool = False
    max_duration: int = 300
    result: dict | None = None


# ③ 工具节点
def start_record(state: AVState):
    """
    state: 当前的图状态（LangGraph 自动传入）
    返回值: 需要写回 state 的增量
    """
    msg = start_av_recording()     # ⚠️ 不接受参数的底层函数
    return {"recording_started": True, "start_msg": msg}

def stop_record(state: AVState):
    print("🟥 stop_record 节点被触发")
    result = stop_av_recording()
    return {"result": result}


import keyboard   # pip install keyboard

def wait_for_stop(state: AVState):
    print("🟡 录制中，按 Enter 停止 ...")
    t0 = time.time()
    while True:
        if keyboard.is_pressed("enter"):
            state["external_stop_signal"] = True
            print("🔴 按键停止")
            break
        if time.time() - t0 > state.get("max_duration", 300):
            print("⏰ 超时自动停止")
            break
        time.sleep(0.1)
    return {}


# ⑤ 构建子图
g = StateGraph(AVState)
g.add_node("start_record", start_record)
g.add_node("wait", wait_for_stop)
g.add_node("stop_record", stop_record)

g.add_edge(START, "start_record")
g.add_edge("start_record", "wait")
g.add_edge("wait", "stop_record")
g.add_edge("stop_record", END)

av_subgraph = g.compile()


# ----------------------------------------------------------------------
# ⑥ Tool 包装器：同时支持 dict 位置参数 & 关键字参数
def _wrapper(*args, **kwargs):
    """
    允许以下两种调用：
        av_interview_tool.invoke({"external_stop_signal": True})
        av_interview_tool.invoke(external_stop_signal=True)
    都会被整理为一个 state dict 传给子图
    """
    if len(args) == 1 and isinstance(args[0], dict):
        state = args[0]
    else:
        # 关键字或无参
        state = kwargs
    # 默认超时 300 s，可通过传 max_duration 覆盖
    return av_subgraph.invoke(state)


av_interview_tool = Tool(
    name="av_interview_tool",
    description=(
        "一次性完成面试的音视频采集与分析，"
        "返回 {'audio_dir','video_dir','audio_summary','video_summary'}"
    ),
    func=_wrapper,
)
# ----------------------------------------------------------------------


# ⑦ 快速自测：10 秒后自动停止
if __name__ == "__main__":
    # 状态 dict，稍后 stop_later 线程会把 external_stop_signal 写进去
    shared_state: dict = {"max_duration": 30}

    def stop_later(sec: int = 10):
        time.sleep(sec)
        shared_state["external_stop_signal"] = True
        print(f"⛔️ {sec} 秒到，自动发送 stop 信号")

    threading.Thread(target=stop_later, daemon=True).start()

    res = av_interview_tool.invoke(shared_state)
    print("✅ 完成，返回结果：")
    print(res)
