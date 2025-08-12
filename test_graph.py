# test_graph.py
"""测试graph流程的简化版本"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph.graph import InterviewState, analyze_interview_performance
from agents.analysis_agent import InterviewAnalysisInput

def test_analysis_node():
    """测试分析节点"""
    print("测试面试分析节点...")
    
    # 模拟面试状态
    mock_state: InterviewState = {
        "resume": "张三，计算机专业，3年Python开发经验",
        "round": 3,
        "should_continue": False,
        "messages": [],
        "qa_pairs": [
            ("请自我介绍一下", "我叫张三，有3年的Python开发经验"),
            ("你最擅长什么技术", "我最擅长Python后端开发和数据库优化")
        ],
        "audio_summaries": ["语音清晰流畅", "表达逻辑性强"],
        "video_summaries": ["表情自然", "肢体语言得当"],
        "max_rounds": 3,
        "interview_completed": True,
        "analysis_result": {},
        "structured_results": []
    }
    
    try:
        # 调用分析节点
        result = analyze_interview_performance(mock_state)
        
        print("分析节点执行成功!")
        analysis = result.get("analysis_result", {})
        
        if analysis:
            print(f"总体评分: {analysis.get('overall_score', 'N/A')}")
            print(f"技术能力: {analysis.get('technical_competency', 'N/A')}")
            print(f"沟通能力: {analysis.get('communication_skills', 'N/A')}")
            
            strengths = analysis.get('strengths', [])
            if strengths:
                print(f"优势: {', '.join(strengths)}")
                
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"建议: {', '.join(recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"分析节点测试失败: {e}")
        return False

def test_route_logic():
    """测试路由逻辑"""
    print("\n测试路由逻辑...")
    
    from graph.graph import route_after_tools
    
    # 测试继续面试的路由
    state1 = {"should_continue": True, "interview_completed": False}
    route1 = route_after_tools(state1)
    print(f"继续面试路由: {route1}")
    
    # 测试面试结束的路由
    state2 = {"should_continue": False, "interview_completed": True}
    route2 = route_after_tools(state2)
    print(f"面试结束路由: {route2}")
    
    # 测试直接结束的路由
    state3 = {"should_continue": False, "interview_completed": False}
    route3 = route_after_tools(state3)
    print(f"直接结束路由: {route3}")
    
    expected_routes = ["process_results", "analyze_performance", "__end__"]
    actual_routes = [route1, route2, route3]
    
    if actual_routes == expected_routes:
        print("路由逻辑测试通过!")
        return True
    else:
        print(f"路由逻辑测试失败: 期望 {expected_routes}, 实际 {actual_routes}")
        return False

if __name__ == "__main__":
    print("开始测试graph流程...\n")
    
    success_count = 0
    total_tests = 2
    
    if test_route_logic():
        success_count += 1
    
    if test_analysis_node():
        success_count += 1
    
    print(f"\n测试完成: {success_count}/{total_tests} 成功")
    
    if success_count == total_tests:
        print("graph流程测试通过! 新的面试分析系统已就绪")
    else:
        print("部分测试失败，请检查错误")