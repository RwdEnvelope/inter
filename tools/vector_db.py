# tools/vector_db.py
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datetime import datetime

class VectorDatabase:
    def __init__(self, db_path: str = "data/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化向量模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2的向量维度
        
        # 初始化FAISS索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
        
        # 存储文档信息
        self.documents = []
        self.doc_metadata = []
        
        # 加载已存在的数据库
        self._load_database()
    
    def _load_database(self):
        """加载已存在的向量数据库"""
        index_path = self.db_path / "index.faiss"
        docs_path = self.db_path / "documents.pkl"
        meta_path = self.db_path / "metadata.pkl"
        
        if index_path.exists() and docs_path.exists() and meta_path.exists():
            try:
                # 加载FAISS索引
                self.index = faiss.read_index(str(index_path))
                
                # 加载文档和元数据
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(meta_path, 'rb') as f:
                    self.doc_metadata = pickle.load(f)
                    
                print(f"加载向量数据库: {len(self.documents)} 个文档")
            except Exception as e:
                print(f"加载数据库失败: {e}")
                self._initialize_empty_db()
        else:
            self._initialize_empty_db()
    
    def _initialize_empty_db(self):
        """初始化空数据库"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.doc_metadata = []
    
    def _save_database(self):
        """保存向量数据库"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, str(self.db_path / "index.faiss"))
            
            # 保存文档和元数据
            with open(self.db_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            with open(self.db_path / "metadata.pkl", 'wb') as f:
                pickle.dump(self.doc_metadata, f)
                
            print(f"保存向量数据库: {len(self.documents)} 个文档")
        except Exception as e:
            print(f"保存数据库失败: {e}")
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档到向量数据库"""
        if not text.strip():
            return ""
            
        # 生成文档ID
        doc_id = hashlib.md5(text.encode()).hexdigest()
        
        # 检查是否已存在
        for i, meta in enumerate(self.doc_metadata):
            if meta.get("doc_id") == doc_id:
                print(f"文档已存在: {doc_id[:8]}")
                return doc_id
        
        # 生成向量
        vector = self.model.encode([text])
        vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)  # 归一化
        
        # 添加到索引
        self.index.add(vector.astype('float32'))
        
        # 存储文档和元数据
        self.documents.append(text)
        meta_dict = {
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat(),
            "source": metadata.get("source", "unknown") if metadata else "unknown",
            "type": metadata.get("type", "text") if metadata else "text"
        }
        if metadata:
            meta_dict.update(metadata)
        self.doc_metadata.append(meta_dict)
        
        # 保存数据库
        self._save_database()
        
        print(f"添加文档: {doc_id[:8]} - {text[:50]}...")
        return doc_id
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if self.index.ntotal == 0:
            return []
        
        # 生成查询向量
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        # 搜索
        scores, indices = self.index.search(query_vector.astype('float32'), min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.doc_metadata[idx],
                    "score": float(score),
                    "doc_id": self.doc_metadata[idx]["doc_id"]
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "last_updated": max([meta.get("timestamp", "") for meta in self.doc_metadata]) if self.doc_metadata else None
        }

# 全局向量数据库实例
vector_db = VectorDatabase()