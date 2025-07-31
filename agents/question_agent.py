import os
import random
from typing import Annotated, TypedDict, List
from openai import OpenAI
from langchain_core.messages import AnyMessage,AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

# 初始化 Qwen 客户端（兼容 OpenAI 接口）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ✅ 状态结构：LangGraph 自动拼接消息历史
class AgentState(TypedDict):
    resume: str
    messages: Annotated[List[AnyMessage], add_messages]
    should_continue: bool   # ⬅️ 新增字段

# 工具函数：从大模型返回中提取多个问题（假设每行一个）
def extract_questions(text: str) -> List[str]:
    return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]

# 主函数：question_agent
def question_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    resume = state["resume"]
    round_count = sum(1 for m in messages if m.type == "ai")
    

    # 首轮固定提问
    if round_count == 0:
        return {
            "resume": resume,
            "should_continue": True,
            "messages": [AIMessage(content="请先自我介绍一下。")]
        }


    # 将 LangGraph 消息转换为 OpenAI 格式
    # 提取最后一轮用户输入
    last_user = next((m.content for m in reversed(messages) if m.type == "human"), "")

    # 添加 prompt，引导模型生成多个追问
    prompt = (
        f"以下是用户简历：\n{resume}\n\n"
        f"以下是用户上一轮回答：\n{last_user}\n\n"
        f"请你作为中文面试官，生成2个提问(可以根据回答追问，可以根据简历提新的提问)，每个一行："
        f"如果需要，生成一个新的追问，否则返回“结束”两个字。"
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个专业中文面试官，擅长基于简历和历史提问。"},
            {"role": "user", "content": prompt}
        ]
    )

    reply = completion.choices[0].message.content.strip()

    # 模型判断是否继续
    if reply == "结束" or "没有更多问题" in reply:
        return {
            "resume": resume,
            "should_continue": False,
            "messages": [AIMessage(content="谢谢你的回答，我这边没有其他问题了。")]
        }

    return {
        "resume": resume,
        "should_continue": True,
        "messages": [AIMessage(content=reply)]
    }