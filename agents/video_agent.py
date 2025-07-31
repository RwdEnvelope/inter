import dashscope
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import os

# 加载 API Key
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class VideoAnalysisAgent:
    def __init__(self, model: str = "qwen-vl-plus"):
        self.model = model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        video_path = state["video_path"]
        prompt = (
            "请分析该视频中的人物行为、面部表情、情绪状态、动作姿态"
            "简要描述面试者的肢体语言和情绪表现，不需要逐帧分析。"
            "用连三句话描述动作表情，推测情绪即可，不用很长"
        )

        path = Path(video_path).resolve()
        print(path)
        if not path.exists():
            return {"error": f"找不到视频文件: {path}"}

        # DashScope 支持 video 分析的模型必须支持 video 文件
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": str(path)},
                    {"text": prompt}
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=messages,
            result_format="message"
        )

        try:
            content = response["output"]["choices"][0]["message"]["content"]
        except Exception as e:
            return {"error": f"模型响应解析失败: {e}", "raw_response": response}

        description = []
        for item in content:
            if "text" in item:
                description.append(item["text"].strip())

        return {
            **state,
            "video_analysis": "\n".join(description)
        }
    
class VideoState(dict):
    video_path: str
    video_analysis: str

def build_video_graph():
    builder = StateGraph(VideoState)

    builder.add_node("VideoAnalysis", VideoAnalysisAgent())
    builder.add_edge(START, "VideoAnalysis")
    builder.add_edge("VideoAnalysis", END)

    return builder.compile()
