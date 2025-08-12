# debug_analysis.py
"""调试分析系统"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.analysis_agent import analysis_agent, InterviewAnalysisInput

def debug_analysis():
    """调试分析过程"""
    
    # 使用简单的测试数据
    test_data: InterviewAnalysisInput = {
        "resume": "测试候选人",
        "qa_pairs": [
            ("测试问题", "测试回答")
        ],
        "audio_summaries": ["测试音频"],
        "video_summaries": ["测试视频"],
        "structured_results": []
    }
    
    print("开始调试分析过程...")
    
    try:
        # 直接调用基础分析
        print("1. 测试基础分析...")
        result = analysis_agent._create_basic_analysis(test_data)
        print(f"基础分析成功: 总分 {result.overall_score}")
        
        # 测试动态评分
        print("\n2. 测试动态评分...")
        scores = analysis_agent._calculate_dynamic_scores(test_data, "")
        print(f"动态评分成功: {scores}")
        
        # 测试洞察提取
        print("\n3. 测试洞察提取...")
        insights = analysis_agent._extract_insights_from_content(test_data, "")
        print(f"洞察提取成功: {insights}")
        
        # 完整分析测试
        print("\n4. 测试完整分析...")
        full_result = analysis_agent.analyze_interview(test_data)
        print(f"完整分析成功: 总分 {full_result.overall_score}")
        print(f"优势: {full_result.strengths}")
        print(f"建议: {full_result.recommendations}")
        
    except Exception as e:
        print(f"调试过程遇到错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_analysis()