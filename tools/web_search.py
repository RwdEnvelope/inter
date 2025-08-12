# tools/web_search.py
import requests
import json
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import time
from urllib.parse import quote_plus
from tools.vector_db import vector_db

class WebSearcher:
    def __init__(self):
        # 可以配置多个搜索引擎
        self.search_engines = {
            "duckduckgo": "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
            "searx": "https://searx.be/search?q={}&format=json&categories=general"
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """使用DuckDuckGo搜索"""
        try:
            url = self.search_engines["duckduckgo"].format(quote_plus(query))
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # DuckDuckGo即时答案
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("AbstractText", ""),
                        "url": data.get("AbstractURL", ""),
                        "content": data.get("Abstract", ""),
                        "source": "duckduckgo_instant"
                    })
                
                # 相关主题
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text", "")[:100],
                            "url": topic.get("FirstURL", ""),
                            "content": topic.get("Text", ""),
                            "source": "duckduckgo_related"
                        })
                
                return results[:max_results]
            
        except Exception as e:
            print(f"❌ DuckDuckGo搜索失败: {e}")
        
        return []
    
    def search_web_fallback(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """备用搜索方法"""
        # 这里可以集成其他搜索API，比如Google Custom Search
        # 暂时返回模拟结果
        return [{
            "title": f"关于'{query}'的相关信息",
            "url": "https://example.com",
            "content": f"这是关于{query}的一般性信息。建议查阅相关专业资料获得更准确的答案。",
            "source": "fallback"
        }]
    
    def extract_page_content(self, url: str) -> str:
        """提取网页内容"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 移除脚本和样式
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 提取文本
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:2000]  # 限制长度
        except Exception as e:
            print(f"❌ 提取网页内容失败: {e}")
        
        return ""

# 全局搜索实例
web_searcher = WebSearcher()

@tool  
def search_and_save_tool(query: str, max_results: int = 3) -> str:
    """
    搜索网络信息并保存到向量数据库
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
    
    Returns:
        搜索结果摘要
    """
    print(f"开始搜索: {query}")
    
    # 先在向量数据库中搜索
    existing_results = vector_db.search(query, top_k=3, threshold=0.8)
    if existing_results:
        print(f"在数据库中找到 {len(existing_results)} 个相关结果")
        best_result = existing_results[0]
        return f"在知识库中找到相关信息：{best_result['document'][:300]}..."
    
    # 数据库中没有找到，进行网络搜索
    print("在网络中搜索...")
    search_results = web_searcher.search_duckduckgo(query, max_results)
    
    if not search_results:
        search_results = web_searcher.search_web_fallback(query, max_results)
    
    # 保存搜索结果到数据库
    saved_docs = []
    for result in search_results:
        content = result.get("content", "")
        if content.strip():
            # 添加到向量数据库
            doc_id = vector_db.add_document(
                text=content,
                metadata={
                    "source": "web_search",
                    "query": query,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "search_engine": result.get("source", "unknown")
                }
            )
            saved_docs.append(content[:100])
    
    if saved_docs:
        summary = f"搜索到 {len(saved_docs)} 个结果并保存到知识库。主要内容：\n"
        for i, doc in enumerate(saved_docs, 1):
            summary += f"{i}. {doc}...\n"
        return summary
    else:
        return f"未找到关于'{query}'的有用信息。"

@tool
def query_knowledge_base_tool(query: str, threshold: float = 0.7) -> str:
    """
    查询向量数据库中的知识
    
    Args:
        query: 查询内容
        threshold: 相似度阈值
    
    Returns:
        查询结果
    """
    print(f"查询知识库: {query}")
    
    results = vector_db.search(query, top_k=5, threshold=threshold)
    
    if not results:
        return f"在知识库中未找到关于'{query}'的相关信息。"
    
    response = f"在知识库中找到 {len(results)} 个相关结果：\n\n"
    for i, result in enumerate(results, 1):
        response += f"{i}. [相似度: {result['score']:.2f}]\n"
        response += f"   {result['document'][:200]}...\n\n"
    
    return response