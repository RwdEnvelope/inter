# graph/graph.py
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated

# 项目根路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tools.analysis import start_av_recording, stop_av_recording
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage, ToolMessage

# ==== LangGraph ====
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from agents.question_agent import create_question_agent
from graph.av_workflow import av_interview_tool
from agents.analysis_agent import analysis_agent, InterviewAnalysisInput

# 状态定义
class InterviewState(TypedDict):
    resume: str
    round: int
    should_continue: bool
    messages: Annotated[List[AnyMessage], add_messages]
    qa_pairs: List[tuple[str, str]]
    audio_summaries: List[str]
    video_summaries: List[str]
    max_rounds: int
    # 新增字段
    interview_completed: bool
    analysis_result: dict
    structured_results: List[dict]

tools = [av_interview_tool]
tool_node = ToolNode(tools)

def assistant(state: InterviewState) -> InterviewState:
    """面试官助手函数 - 提问并决定是否继续"""
    current_round = state.get("round", 0)
    max_rounds = state.get("max_rounds", 10)
    
    # 第一轮：自我介绍
    if current_round == 0:
        print(f"🎬 开始第 {current_round + 1} 轮面试")
        question = "请先自我介绍一下。"
        
        # ✅ 关键修改：返回工具调用消息而不是普通消息
        tool_call_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "av_interview_tool", 
                "args": {"question": question, "max_duration": 30},
                "id": f"call_{current_round + 1}",
                "type": "tool_call"
            }]
        )
        
        return {
            "messages": [AIMessage(content=question), tool_call_message],
            "round": current_round + 1,
            "should_continue": True,
        }
    
    # 检查是否超过最大轮次
    if current_round >= max_rounds:
        print(f"⏰ 已达到最大轮次 {max_rounds}，结束面试")
        return {
            "messages": [AIMessage(content="感谢您的时间，面试结束。")],
            "round": current_round,
            "should_continue": False,
            "interview_completed": True,
        }
    
    # 其他轮次：调用LLM决策
    try:
        print(f"🎬 开始第 {current_round + 1} 轮面试")
        llm = create_question_agent()
        decision = llm.invoke(state)
        
        print(f"🤖 决策结果: {decision}")
        
        messages = []
        should_continue = False
        
        # 如果有下一个问题，则继续
        if decision.next_question:
            question = decision.next_question
            messages.append(AIMessage(content=question))
            
            # ✅ 关键修改：生成工具调用消息
            tool_call_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": "av_interview_tool", 
                    "args": {"question": question, "max_duration": 60},
                    "id": f"call_{current_round + 1}",
                    "type": "tool_call"
                }]
            )
            messages.append(tool_call_message)
            should_continue = decision.should_round
        else:
            # 没有下一个问题，结束面试
            messages.append(AIMessage(content="感谢您的时间，面试结束。"))
            should_continue = False
        
        return {
            "messages": messages,
            "round": current_round + 1,
            "should_continue": should_continue,



            
            "interview_completed": not should_continue,
        }
        
    except Exception as e:
        print(f"❌ assistant函数错误: {e}")
        # 异常情况下的兜底处理
        question = "请继续介绍您的经验。"
        tool_call_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "av_interview_tool", 
                "args": {"question": question, "max_duration": 60},
                "id": f"call_{current_round + 1}",
                "type": "tool_call"
            }]
        )
        return {
            "messages": [AIMessage(content=question), tool_call_message],
            "round": current_round + 1,
            "should_continue": True,
        }

def process_tool_results(state: InterviewState) -> InterviewState:
    """处理工具执行结果"""
    messages = state.get("messages", [])
    qa_pairs = list(state.get("qa_pairs", []))
    audio_summaries = list(state.get("audio_summaries", []))
    video_summaries = list(state.get("video_summaries", []))
    
    # 查找最新的问题和工具结果
    question = None
    tool_result = None
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            question = msg.content
        elif isinstance(msg, ToolMessage):
            tool_result = msg.content
            break
    
    if question and tool_result:
        try:
            import json
            result_data = json.loads(tool_result)
            
            # 提取音频摘要和视频摘要
            audio_summary = result_data.get("audio_summary", "")
            video_summary = result_data.get("video_summary", "")
            
            if audio_summary:
                audio_summaries.append(audio_summary)
            if video_summary:
                video_summaries.append(video_summary)
            
            # 将问答对加入记录
            qa_pairs.append((question, audio_summary))
            
            print(f"📝 本轮问答: Q: {question[:50]}... A: {audio_summary[:50]}...")
            
        except json.JSONDecodeError as e:
            print(f"❌ 解析工具结果失败: {e}")
    
    return {
        "qa_pairs": qa_pairs,
        "audio_summaries": audio_summaries,
        "video_summaries": video_summaries,
    }

def analyze_interview_performance(state: InterviewState) -> InterviewState:
    """分析面试表现的节点"""
    print("开始面试分析...")
    
    try:
        # 准备分析输入数据
        analysis_input: InterviewAnalysisInput = {
            "resume": state.get("resume", ""),
            "qa_pairs": state.get("qa_pairs", []),
            "audio_summaries": state.get("audio_summaries", []),
            "video_summaries": state.get("video_summaries", []),
            "structured_results": state.get("structured_results", [])
        }
        
        # 执行分析
        analysis_result = analysis_agent.analyze_interview(analysis_input)
        
        # 格式化分析结果为字典
        result_dict = {
            "overall_score": analysis_result.overall_score,
            "technical_competency": analysis_result.technical_competency,
            "communication_skills": analysis_result.communication_skills,
            "problem_solving": analysis_result.problem_solving,
            "strengths": analysis_result.strengths,
            "weaknesses": analysis_result.weaknesses,
            "recommendations": analysis_result.recommendations,
            "key_insights": analysis_result.key_insights,
            "behavioral_analysis": analysis_result.behavioral_analysis,
            "detailed_analysis": analysis_result.detailed_analysis
        }
        
        print("面试分析完成")
        print(f"总体评分: {analysis_result.overall_score}/100")
        print(f"主要优势: {', '.join(analysis_result.strengths[:2])}")
        
        return {
            "analysis_result": result_dict,
            "messages": [AIMessage(content=f"面试分析已完成。候选人总体评分：{analysis_result.overall_score}/100")]
        }
        
    except Exception as e:
        print(f"面试分析失败: {e}")
        return {
            "analysis_result": {
                "error": f"分析失败: {str(e)}",
                "overall_score": 0,
                "status": "failed"
            },
            "messages": [AIMessage(content="面试分析过程中遇到问题，请检查系统设置。")]
        }

def route_after_tools(state: InterviewState) -> str:
    """工具执行后的路由决策"""
    should_continue = state.get("should_continue", False)
    interview_completed = state.get("interview_completed", False)
    
    if should_continue:
        route = "process_results"
    elif interview_completed:
        route = "analyze_performance"
    else:
        route = "__end__"
        
    print(f"工具后路由: should_continue={should_continue}, completed={interview_completed} -> {route}")
    return route

def route_after_analysis(state: InterviewState) -> str:
    """面试分析后的路由决策"""
    print("分析完成，结束流程")
    return "__end__"

# ✅ 构建图 - 增强版本
g = StateGraph(InterviewState)

# 添加节点
g.add_node("assistant", assistant)
g.add_node("tools", tool_node)
g.add_node("process_results", process_tool_results)
g.add_node("analyze_performance", analyze_interview_performance)

# 添加边
g.add_edge(START, "assistant")

# ✅ 关键修改：assistant有工具调用时自动去tools
g.add_edge("assistant", "tools")

# ✅ tools执行完后的条件路由
g.add_conditional_edges("tools", route_after_tools, {
    "process_results": "process_results",
    "analyze_performance": "analyze_performance",
    "__end__": END
})

# ✅ 处理完结果后回到assistant继续下一轮
g.add_edge("process_results", "assistant")

# ✅ 分析完成后结束
g.add_edge("analyze_performance", END)

print("图结构验证:")
print(f"节点: {list(g.nodes.keys())}")

memory = MemorySaver()
interview_graph = g.compile(checkpointer=memory)

print("图编译成功!")

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    sample_resume = """姓名：Alice
学历：计算机科学本科
经历：阿里巴巴推荐系统实习 6 个月
技能：Python / PyTorch"""

    init_state: InterviewState = {
        "resume": sample_resume,
        "round": 0,
        "max_rounds": 3,  # 减少轮次便于测试
        "messages": [],
        "should_continue": True,
        "qa_pairs": [],
        "audio_summaries": [],
        "video_summaries": [],
        # 新增字段
        "interview_completed": False,
        "analysis_result": {},
        "structured_results": [],
    }

    print("🎯 开始智能面试流程...\n")
    print("📋 新流程: START → assistant → tools → process_results → assistant → ... → analyze_performance → END")
    print("🔍 增加功能: 面试分析 + 向量数据库 + 网络搜索")
    print("=" * 80)
    
    final = None
    step_count = 0
    for chunk in interview_graph.stream(init_state, config=config, stream_mode="values"):
        final = chunk
        step_count += 1
        
        print(f"\n🔄 步骤 {step_count}")
        
        if chunk.get("messages"):
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"🤖 面试官 [轮次{chunk.get('round', 0)}]: {last_msg.content}")
            elif isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                print(f"🔧 调用工具: {last_msg.tool_calls[0]['name']}")
            elif isinstance(last_msg, ToolMessage):
                print(f"🔧 工具执行完成 [轮次{chunk.get('round', 0)}]")
                result_preview = last_msg.content[:100] + "..." if len(last_msg.content) > 100 else last_msg.content
                print(f"🔧 工具结果预览: {result_preview}")
        
        should_continue = chunk.get("should_continue", False)
        print(f"📍 继续面试: {'是' if should_continue else '否'}")
        print("-" * 40)

    print("\n=== 🎉 面试结束，完整回顾 ===")
    qa_pairs = final.get("qa_pairs", [])
    if qa_pairs:
        for i, (q, a) in enumerate(qa_pairs, 1):
            print(f"\n【第{i}问】{q}")
            print(f"【回答】{a}")
    else:
        print("暂无问答记录")

    print(f"\n📝 音频摘要: {len(final.get('audio_summaries', []))} 条")
    audio_summaries = final.get("audio_summaries", [])
    for i, summary in enumerate(audio_summaries, 1):
        if summary:
            print(f"  {i}. {summary[:100]}...")

    print(f"\n🎞️ 视频摘要: {len(final.get('video_summaries', []))} 条")
    video_summaries = final.get("video_summaries", [])
    for i, summary in enumerate(video_summaries, 1):
        if summary:
            print(f"  {i}. {summary[:100]}...")
    
    # 显示面试分析结果
    analysis_result = final.get("analysis_result", {})
    if analysis_result and not analysis_result.get("error"):
        print(f"\n🎯 === 面试分析报告 ===")
        print(f"📊 总体评分: {analysis_result.get('overall_score', 0)}/100")
        print(f"💻 技术能力: {analysis_result.get('technical_competency', 0)}/100")
        print(f"🗣️ 沟通能力: {analysis_result.get('communication_skills', 0)}/100")
        print(f"🧩 问题解决: {analysis_result.get('problem_solving', 0)}/100")
        
        strengths = analysis_result.get("strengths", [])
        if strengths:
            print(f"\n✅ 优势: {', '.join(strengths)}")
            
        weaknesses = analysis_result.get("weaknesses", [])
        if weaknesses:
            print(f"⚠️ 待改进: {', '.join(weaknesses)}")
            
        recommendations = analysis_result.get("recommendations", [])
        if recommendations:
            print(f"💡 建议: {', '.join(recommendations)}")
            
        behavioral = analysis_result.get("behavioral_analysis", "")
        if behavioral:
            print(f"\n🎭 行为分析: {behavioral}")
            
    elif analysis_result.get("error"):
        print(f"\n❌ 分析遇到问题: {analysis_result.get('error')}")
    else:
        print(f"\n📝 未生成分析报告")