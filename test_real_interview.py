# test_real_interview.py
"""模拟真实面试数据测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph.graph import analyze_interview_performance, InterviewState

def test_real_interview():
    """测试真实面试数据"""
    
    # 模拟graph中的真实状态数据
    mock_state: InterviewState = {
        "resume": "姓名：Alice\n学历：计算机科学本科\n经历：阿里巴巴推荐系统实习 6 个月\n技能：Python / PyTorch",
        "round": 3,
        "should_continue": False,
        "messages": [],
        "qa_pairs": [
            ("请先自我介绍一下。", "语音内容: 我叫Alice，计算机科学本科毕业，在阿里巴巴推荐系统实习了6个月，主要使用Python和PyTorch进行机器学习模型开发\n情绪分析: 自信度:0.8"),
            ("请继续介绍您的经验。", "语音内容: 在实习期间，我参与了推荐算法的优化工作，使用深度学习改进了用户画像模型，将点击率提升了15%\n情绪分析: 专业度:0.9"),
            ("请继续介绍您的经验。", "语音内容: 我还负责过数据管道的搭建，熟悉Spark和Kafka等大数据技术，能够处理TB级别的用户行为数据\n情绪分析: 技术能力:0.85")
        ],
        "audio_summaries": ["语音清晰，表达流畅，情绪稳定", "专业术语使用准确，逻辑清晰", "回答完整，技术深度较好"],
        "video_summaries": ["面部表情自然，肢体语言得当，注意力集中", "保持良好的眼神交流", "整体表现专业"],
        "max_rounds": 3,
        "interview_completed": True,
        "analysis_result": {},
        "structured_results": []
    }
    
    print("测试真实面试数据分析...")
    print(f"问答对数量: {len(mock_state['qa_pairs'])}")
    print(f"音频摘要数量: {len(mock_state['audio_summaries'])}")
    print(f"视频摘要数量: {len(mock_state['video_summaries'])}")
    
    try:
        # 调用实际的分析节点
        result = analyze_interview_performance(mock_state)
        
        analysis = result.get("analysis_result", {})
        if analysis and not analysis.get("error"):
            print(f"\n分析成功!")
            print(f"总体评分: {analysis.get('overall_score')}/100")
            print(f"技术能力: {analysis.get('technical_competency')}/100")
            print(f"沟通能力: {analysis.get('communication_skills')}/100")
            print(f"问题解决: {analysis.get('problem_solving')}/100")
            
            strengths = analysis.get('strengths', [])
            if strengths:
                print(f"优势: {', '.join(strengths)}")
                
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"建议: {', '.join(recommendations)}")
                
            # 检查评分方法
            detailed = analysis.get('detailed_analysis', {})
            scoring_method = detailed.get('scoring_method', 'unknown')
            print(f"评分方法: {scoring_method}")
            
        else:
            print(f"分析失败: {analysis.get('error', 'unknown error')}")
            
    except Exception as e:
        print(f"测试过程遇到错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_interview()