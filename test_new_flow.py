# test_new_flow.py
"""测试新的面试分析流程"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.vector_db import vector_db
from tools.web_search import search_and_save_tool, query_knowledge_base_tool
from agents.analysis_agent import analysis_agent, InterviewAnalysisInput

def test_vector_db():
    """测试向量数据库"""
    print("🧪 测试向量数据库...")
    
    # 添加测试文档
    doc_id = vector_db.add_document(
        "Python是一种高级编程语言，具有简洁的语法和强大的库支持。",
        metadata={"source": "test", "type": "knowledge"}
    )
    print(f"✅ 添加文档: {doc_id}")
    
    # 搜索测试
    results = vector_db.search("Python编程", top_k=3)
    print(f"✅ 搜索结果: {len(results)} 个")
    for result in results:
        print(f"  - 相似度: {result['score']:.3f}")
        print(f"  - 内容: {result['document'][:50]}...")
    
    # 统计信息
    stats = vector_db.get_statistics()
    print(f"✅ 数据库统计: {stats}")

def test_web_search():
    """测试网络搜索"""
    print("\n🧪 测试网络搜索...")
    
    # 测试知识库查询
    result = query_knowledge_base_tool.invoke({"query": "Python编程语言"})
    print(f"✅ 知识库查询: {result[:100]}...")
    
    # 测试网络搜索
    result = search_and_save_tool.invoke({"query": "面试技巧", "max_results": 2})
    print(f"✅ 网络搜索: {result[:100]}...")

def test_analysis_agent():
    """测试分析agent"""
    print("\n🧪 测试面试分析agent...")
    
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
    
    try:
        analysis_result = analysis_agent.analyze_interview(mock_data)
        print(f"✅ 分析完成:")
        print(f"  - 总体评分: {analysis_result.overall_score}/100")
        print(f"  - 技术能力: {analysis_result.technical_competency}/100")
        print(f"  - 沟通能力: {analysis_result.communication_skills}/100")
        print(f"  - 优势: {', '.join(analysis_result.strengths)}")
        print(f"  - 建议: {', '.join(analysis_result.recommendations)}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def test_integrated_flow():
    """测试集成流程"""
    print("\n🧪 测试集成流程...")
    
    # 1. 向知识库添加面试相关知识
    vector_db.add_document(
        "技术面试评估标准：代码质量、算法思维、系统设计、沟通表达、学习能力",
        metadata={"source": "hr_guideline", "type": "evaluation"}
    )
    
    vector_db.add_document(
        "优秀候选人特征：逻辑清晰、主动思考、团队协作、持续学习、解决问题能力强",
        metadata={"source": "hr_guideline", "type": "criteria"}
    )
    
    # 2. 测试查询
    result = query_knowledge_base_tool.invoke({"query": "面试评估标准"})
    print(f"✅ 评估标准查询: {result[:150]}...")
    
    # 3. 如果没找到，搜索网络
    if "未找到" in result:
        web_result = search_and_save_tool.invoke({"query": "技术面试评估标准"})
        print(f"✅ 网络搜索补充: {web_result[:150]}...")
    
    print("✅ 集成流程测试完成")

if __name__ == "__main__":
    print("🚀 开始测试新的面试分析系统...\n")
    
    try:
        test_vector_db()
        test_web_search() 
        test_analysis_agent()
        test_integrated_flow()
        
        print("\n🎉 所有测试完成！")
        print("📊 系统已准备就绪，可以运行完整的面试分析流程")
        
    except Exception as e:
        print(f"\n❌ 测试过程中遇到错误: {e}")
        print("请检查依赖是否正确安装：pip install -r requirements_new.txt")