# test_question_agent.py
from agents.question_agent import question_agent, AgentState
from langchain_core.messages import HumanMessage

import time

# 准备多个候选人的简历
user_profiles = {
    "Alice": "姓名：Alice\n学历：计算机科学本科\n经历：在阿里巴巴实习6个月，参与推荐系统的工程开发。",
    "Bob": "姓名：Bob\n学历：人工智能专业研究生\n经历：曾获数学建模国赛一等奖，熟悉PyTorch与深度学习。",
    "Charlie": "姓名：Charlie\n学历：软件工程大三\n经历：作为主要开发者完成过一个校园电商平台，熟悉前后端开发。"
}

# 每个用户的状态
user_states: dict[str, AgentState] = {
    name: {
        "resume": resume,
        "messages": []
    } for name, resume in user_profiles.items()
}

# 每轮都调用 agent，用户模拟回答
def simulate_conversation(username: str, rounds: int = 3):
    print(f"\n========== {username} 的面试开始 ==========")
    state = user_states[username]

    for i in range(rounds):
        # AI 提问
        state = question_agent(state)
        ai_msg = state["messages"][-1].content
        print(f"🤖 面试官：{ai_msg}")

        # 用户回答（模拟/输入）
        user_reply = input(f"🧑‍💼 {username} 的回答：")
        state["messages"].append(HumanMessage(content=user_reply))

        # 等待一点时间（模拟节奏）
        time.sleep(0.5)

    print(f"✅ {username} 的面试结束\n")
    user_states[username] = state  # 保存状态

# 测试所有人
if __name__ == "__main__":
    for username in user_profiles.keys():
        simulate_conversation(username)