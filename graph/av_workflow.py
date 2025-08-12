# graph/av_workflow.py
from __future__ import annotations
import sys
import threading
import time
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any, Annotated
import operator

# 项目根路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool  # ✅ 使用 @tool 装饰器
from langchain_core.messages import AnyMessage

# 假设这些函数已实现
from tools.analysis import start_av_recording, stop_av_recording

class AVState(TypedDict, total=False):
    """子图的运行状态"""
    messages: Annotated[list[Any], operator.add]
    recording_started: bool
    external_stop_signal: bool
    max_duration: int

def start_record(state: AVState) -> AVState:
    """启动音视频录制（非阻塞）"""
    start_av_recording()
    return {"recording_started": True}

def stop_record(state: AVState) -> AVState:
    """停止录制并收集结果"""
    print("🟥 停止录制并整理结果")
    print("🟥 stop_record 节点被触发")
    result = stop_av_recording()
    return {"messages": [result]}

def wait_for_stop(state: AVState) -> AVState:
    """等待停止信号"""
    print("🟡 录制中，按 Enter 停止 ...")
    t0 = time.time()
    max_duration = state.get("max_duration", 300)
    
    while True:
        # 检查外部信号
        if state.get("external_stop_signal", False):
            print("🔴 外部停止信号")
            return {}
        
        # 检查键盘（如果可用）
        try:
            import keyboard
            if keyboard.is_pressed("enter"):
                print("🔴 按键停止")
                return {"external_stop_signal": True}
        except:
            pass  # keyboard 模块不可用时忽略
        
        # 检查超时
        if time.time() - t0 > max_duration:
            print("⏰ 超时自动停止")
            return {"external_stop_signal": True}
            
        time.sleep(0.1)

# 构建子图
g = StateGraph(AVState)
g.add_node("start_record", start_record)
g.add_node("wait", wait_for_stop)
g.add_node("stop_record", stop_record)

g.add_edge(START, "start_record")
g.add_edge("start_record", "wait")
g.add_edge("wait", "stop_record")
g.add_edge("stop_record", END)

av_subgraph = g.compile()

# ✅ 使用 @tool 装饰器重新定义工具
@tool
def av_interview_tool(question: str = "", max_duration: int = 300) -> str:
    """
    一次性完成面试的音视频采集与分析，返回分析结果
    
    Args:
        question: 面试问题（可选）
        max_duration: 最大录制时长（秒），默认300秒
    
    Returns:
        包含音视频分析结果的JSON字符串
    """
    print(f"🎬 工具被调用: question='{question}', max_duration={max_duration}")
    
    # 设置状态
    state = {
        "max_duration": max_duration,
        "external_stop_signal": False,
        "messages": [],
        "recording_started": False
    }
    
    try:
        # 执行子图
        result = av_subgraph.invoke(state)
        
        if 'messages' in result and result['messages']:
            actual_result = result['messages'][0]
            import json
            return json.dumps(actual_result, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "audio_summary": "音视频录制完成，但未获得分析结果",
                "video_summary": "",
                "audio_dir": "",
                "video_dir": ""
            }, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 录制工具执行失败: {e}")
        import json
        return json.dumps({
            "error": f"录制工具执行失败: {str(e)}",
            "audio_summary": "",
            "video_summary": "",
            "audio_dir": "",
            "video_dir": ""
        }, ensure_ascii=False, indent=2)

# 测试代码
if __name__ == "__main__":
    print("🧪 测试工具...")
    
    # 模拟自动停止
    shared_state = {"max_duration": 10}
    
    def stop_later(sec: int = 5):
        time.sleep(sec)
        shared_state["external_stop_signal"] = True
        print(f"⛔️ {sec} 秒到，自动发送 stop 信号")

    threading.Thread(target=stop_later, daemon=True).start()
    
    # 测试工具调用
    res = av_interview_tool.invoke({"question": "请自我介绍", "max_duration": 10})
    print("✅ 完成，返回结果：")
    print(res)