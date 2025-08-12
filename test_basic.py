# -*- coding: utf-8 -*-
# test_basic.py
"""测试基础功能（不依赖OpenAI API）"""

import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_vector_db_only():
    """仅测试向量数据库功能"""
    print("测试向量数据库...")
    
    try:
        from tools.vector_db import vector_db
        
        # 添加测试文档
        doc_id = vector_db.add_document(
            "Python是一种高级编程语言，具有简洁的语法和强大的库支持。",
            metadata={"source": "test", "type": "knowledge"}
        )
        print(f"添加文档: {doc_id[:8]}")
        
        # 搜索测试
        results = vector_db.search("Python编程", top_k=3)
        print(f"搜索结果: {len(results)} 个")
        for result in results:
            print(f"  - 相似度: {result['score']:.3f}")
            print(f"  - 内容: {result['document'][:50]}...")
        
        # 统计信息
        stats = vector_db.get_statistics()
        print(f"数据库统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"向量数据库测试失败: {e}")
        return False

def test_basic_analysis():
    """测试基础分析功能"""
    print("\n🧪 测试基础分析功能...")
    
    try:
        from agents.analysis_agent import analysis_agent, InterviewAnalysisInput
        
        # 模拟面试数据
        mock_data: InterviewAnalysisInput = {
            "resume": "张三，计算机专业，3年Python开发经验",
            "qa_pairs": [
                ("请自我介绍一下", "我叫张三，有3年的Python开发经验，熟悉Django和Flask框架"),
                ("描述一下你最有挑战的项目", "我开发了一个电商系统，处理了高并发和数据一致性问题")
            ],
            "audio_summaries": ["语音清晰，表达流畅，情绪稳定"],
            "video_summaries": ["面部表情自然，肢体语言得当，注意力集中"],
            "structured_results": []
        }
        
        # 执行分析
        analysis_result = analysis_agent.analyze_interview(mock_data)
        print(f"✅ 分析完成:")
        print(f"  - 总体评分: {analysis_result.overall_score}/100")
        print(f"  - 技术能力: {analysis_result.technical_competency}/100")
        print(f"  - 沟通能力: {analysis_result.communication_skills}/100")
        print(f"  - 优势: {', '.join(analysis_result.strengths)}")
        print(f"  - 建议: {', '.join(analysis_result.recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础分析测试失败: {e}")
        return False

def test_web_search_basic():
    """测试基础搜索功能"""
    print("\n🧪 测试基础搜索功能...")
    
    try:
        from tools.web_search import web_searcher
        
        # 测试DuckDuckGo搜索
        results = web_searcher.search_duckduckgo("Python programming", max_results=2)
        print(f"✅ DuckDuckGo搜索: {len(results)} 个结果")
        for result in results:
            print(f"  - 标题: {result.get('title', '')[:50]}...")
        
        # 测试备用搜索
        fallback_results = web_searcher.search_web_fallback("面试技巧", max_results=2)
        print(f"✅ 备用搜索: {len(fallback_results)} 个结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 搜索功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始基础功能测试...\n")
    
    success_count = 0
    total_tests = 3
    
    # 测试向量数据库
    if test_vector_db_only():
        success_count += 1
    
    # 测试基础分析
    if test_basic_analysis():
        success_count += 1
    
    # 测试搜索功能
    if test_web_search_basic():
        success_count += 1
    
    print(f"\n📊 测试完成: {success_count}/{total_tests} 成功")
    
    if success_count == total_tests:
        print("🎉 所有基础功能正常！")
        print("💡 要获得完整AI分析功能，请设置OPENAI_API_KEY环境变量")
    else:
        print("⚠️ 部分功能测试失败，请检查错误信息")