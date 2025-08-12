# simple_test.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_imports():
    """测试导入"""
    try:
        print("测试导入向量数据库...")
        from tools.vector_db import vector_db
        print("OK - 向量数据库导入成功")
        
        print("测试导入分析agent...")
        from agents.analysis_agent import analysis_agent
        print("OK - 分析agent导入成功 (基础模式)")
        
        print("测试导入搜索工具...")
        from tools.web_search import query_knowledge_base_tool, search_and_save_tool
        print("OK - 搜索工具导入成功")
        
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False

def test_vector_db():
    """测试向量数据库"""
    try:
        from tools.vector_db import vector_db
        
        # 添加文档
        doc_id = vector_db.add_document("这是一个测试文档")
        print(f"添加文档成功: {doc_id[:8]}")
        
        # 搜索
        results = vector_db.search("测试", top_k=1)
        print(f"搜索成功: 找到 {len(results)} 个结果")
        
        return True
    except Exception as e:
        print(f"向量数据库测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始简单测试...")
    
    if test_imports():
        print("导入测试通过")
        if test_vector_db():
            print("向量数据库测试通过")
            print("基础功能正常!")
        else:
            print("向量数据库测试失败")
    else:
        print("导入测试失败")