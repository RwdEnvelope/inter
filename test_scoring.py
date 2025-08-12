# test_scoring.py
"""测试新的动态评分系统"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.analysis_agent import analysis_agent, InterviewAnalysisInput

def test_scoring_scenarios():
    """测试不同的评分场景"""
    
    # 场景1: 优秀候选人
    print("=== 场景1: 优秀候选人 ===")
    excellent_data: InterviewAnalysisInput = {
        "resume": "李明，5年Python开发经验，阿里巴巴高级工程师",
        "qa_pairs": [
            ("请自我介绍", "我叫李明，有5年Python开发经验，熟悉Django、Flask框架，负责过分布式系统架构设计，在阿里巴巴担任高级工程师，主要负责推荐系统的性能优化和算法改进"),
            ("描述你最有挑战的项目", "我负责的项目是一个高并发的推荐系统，日处理10亿+请求。主要挑战是如何优化算法性能和解决分布式一致性问题。我通过引入缓存策略、优化数据结构、使用微服务架构等方案，最终将响应时间从200ms优化到50ms"),
            ("如何解决团队冲突", "首先分析冲突原因，然后组织团队会议进行开放讨论，最后制定共同认可的解决方案。我认为沟通和理解是解决团队问题的关键")
        ],
        "audio_summaries": ["语音清晰流畅，表达逻辑性强，语调自信稳定"],
        "video_summaries": ["面部表情自然，肢体语言得当，注意力高度集中"],
        "structured_results": []
    }
    
    result1 = analysis_agent.analyze_interview(excellent_data)
    print(f"总体评分: {result1.overall_score}/100")
    print(f"技术能力: {result1.technical_competency}/100") 
    print(f"沟通能力: {result1.communication_skills}/100")
    print(f"问题解决: {result1.problem_solving}/100")
    print(f"优势: {result1.strengths}")
    print(f"建议: {result1.recommendations}")
    
    # 场景2: 一般候选人
    print("\n=== 场景2: 一般候选人 ===")
    average_data: InterviewAnalysisInput = {
        "resume": "张三，2年开发经验",
        "qa_pairs": [
            ("请自我介绍", "我叫张三，做了2年开发"),
            ("你会什么技术", "会Python和Java")
        ],
        "audio_summaries": ["语音一般"],
        "video_summaries": ["表情正常"],
        "structured_results": []
    }
    
    result2 = analysis_agent.analyze_interview(average_data)
    print(f"总体评分: {result2.overall_score}/100")
    print(f"技术能力: {result2.technical_competency}/100")
    print(f"沟通能力: {result2.communication_skills}/100") 
    print(f"问题解决: {result2.problem_solving}/100")
    print(f"优势: {result2.strengths}")
    print(f"劣势: {result2.weaknesses}")
    print(f"建议: {result2.recommendations}")
    
    # 场景3: 表现较差的候选人
    print("\n=== 场景3: 表现较差 ===")
    poor_data: InterviewAnalysisInput = {
        "resume": "王五，无经验",
        "qa_pairs": [
            ("请自我介绍", "我是王五")
        ],
        "audio_summaries": ["语音不清楚，紧张"],
        "video_summaries": ["表情紧张，不适"],
        "structured_results": []
    }
    
    result3 = analysis_agent.analyze_interview(poor_data)
    print(f"总体评分: {result3.overall_score}/100")
    print(f"技术能力: {result3.technical_competency}/100")
    print(f"沟通能力: {result3.communication_skills}/100")
    print(f"问题解决: {result3.problem_solving}/100")
    print(f"优势: {result3.strengths}")
    print(f"劣势: {result3.weaknesses}")
    print(f"建议: {result3.recommendations}")

if __name__ == "__main__":
    print("测试动态评分系统...\n")
    test_scoring_scenarios()
    print("\n测试完成! 评分系统现在会根据实际面试内容动态调整分数。")