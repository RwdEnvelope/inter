# agents/analysis_agent.py
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tools.web_search import search_and_save_tool, query_knowledge_base_tool
from tools.vector_db import vector_db
import json

class InterviewAnalysisInput(TypedDict):
    """面试分析输入"""
    resume: str
    qa_pairs: List[tuple[str, str]]
    audio_summaries: List[str] 
    video_summaries: List[str]
    structured_results: Optional[List[Dict[str, Any]]]

class AnalysisResult(BaseModel):
    """分析结果结构"""
    overall_score: float = Field(description="总体评分 (0-100)")
    technical_competency: float = Field(description="技术能力评分 (0-100)")
    communication_skills: float = Field(description="沟通能力评分 (0-100)")
    problem_solving: float = Field(description="问题解决能力评分 (0-100)")
    
    strengths: List[str] = Field(description="候选人优势")
    weaknesses: List[str] = Field(description="需要改进的方面")
    recommendations: List[str] = Field(description="招聘建议")
    
    key_insights: List[str] = Field(description="关键洞察")
    behavioral_analysis: str = Field(description="行为分析总结")
    
    detailed_analysis: Dict[str, Any] = Field(description="详细分析数据")

class InterviewAnalysisAgent:
    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=4000
            )
            self.llm_available = True
        except Exception as e:
            print(f"LLM初始化失败: {e}")
            print("请设置OPENAI_API_KEY环境变量")
            self.llm = None
            self.llm_available = False
        
        # 工具列表
        self.tools = [query_knowledge_base_tool, search_and_save_tool]
        
        # 绑定工具到LLM
        if self.llm_available:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = None
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深的人力资源专家和技术面试官。你的任务是分析候选人的面试表现，提供专业的评估报告。

分析维度包括：
1. 技术能力：专业知识、问题解决、代码质量等
2. 沟通能力：表达清晰度、逻辑性、互动质量等  
3. 行为表现：自信度、情绪状态、非语言表现等
4. 综合素质：学习能力、适应性、团队合作等

你可以使用以下工具来增强分析：
- query_knowledge_base_tool: 查询已有的面试评估知识库
- search_and_save_tool: 搜索相关的面试评估标准和行业基准

请基于提供的面试数据进行全面分析，并给出专业建议。"""),
            
            ("human", """请分析以下面试数据：

候选人简历：
{resume}

问答对话：
{qa_conversation}

音频分析摘要：
{audio_analysis}

视频分析摘要：  
{video_analysis}

请进行全面的面试分析，包括：
1. 各维度评分 (0-100分)
2. 优势和不足
3. 招聘建议
4. 行为分析
5. 关键洞察

先查询知识库中的相关评估标准，如果没有找到相关信息，请搜索最新的面试评估方法。""")
        ])
    
    def _format_qa_conversation(self, qa_pairs: List[tuple[str, str]]) -> str:
        """格式化问答对话"""
        conversation = ""
        for i, (question, answer) in enumerate(qa_pairs, 1):
            conversation += f"第{i}轮：\n"
            conversation += f"问题：{question}\n"
            conversation += f"回答：{answer}\n\n"
        return conversation
    
    def _extract_behavioral_insights(self, audio_summaries: List[str], video_summaries: List[str]) -> str:
        """提取行为洞察"""
        insights = []
        
        # 音频分析洞察
        if audio_summaries:
            audio_text = " ".join(audio_summaries)
            insights.append(f"语音表现：{audio_text}")
        
        # 视频分析洞察  
        if video_summaries:
            video_text = " ".join(video_summaries)
            insights.append(f"视觉表现：{video_text}")
            
        return " | ".join(insights) if insights else "暂无行为分析数据"
    
    def analyze_interview(self, input_data: InterviewAnalysisInput) -> AnalysisResult:
        """执行面试分析"""
        print("开始面试分析...")
        
        # 检查LLM是否可用
        if not self.llm_available:
            print("LLM不可用，使用基础分析模式")
            try:
                return self._create_basic_analysis(input_data)
            except Exception as e:
                print(f"基础分析模式失败: {e}")
                return self._create_fallback_analysis(input_data)
        
        # 格式化输入数据
        qa_conversation = self._format_qa_conversation(input_data["qa_pairs"])
        audio_analysis = " | ".join(input_data["audio_summaries"])
        video_analysis = " | ".join(input_data["video_summaries"])
        
        # 准备分析提示
        formatted_prompt = self.analysis_prompt.format_messages(
            resume=input_data["resume"],
            qa_conversation=qa_conversation,
            audio_analysis=audio_analysis,
            video_analysis=video_analysis
        )
        
        try:
            # 调用LLM进行分析
            response = self.llm_with_tools.invoke(formatted_prompt)
            
            # 处理工具调用
            final_response = self._handle_tool_calls(response, input_data)
            
            # 解析分析结果
            analysis_result = self._parse_analysis_result(final_response, input_data)
            
            # 保存分析结果到向量数据库
            self._save_analysis_to_db(analysis_result, input_data)
            
            return analysis_result
            
        except Exception as e:
            print(f"面试分析失败: {e}")
            return self._create_fallback_analysis(input_data)
    
    def _handle_tool_calls(self, response, input_data: InterviewAnalysisInput):
        """处理工具调用"""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"处理 {len(response.tool_calls)} 个工具调用")
            
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"调用工具: {tool_name} - {tool_args}")
                
                if tool_name == "query_knowledge_base_tool":
                    result = query_knowledge_base_tool.invoke(tool_args)
                elif tool_name == "search_and_save_tool":
                    result = search_and_save_tool.invoke(tool_args)
                else:
                    result = f"未知工具: {tool_name}"
                
                tool_results.append(f"{tool_name}: {result}")
            
            # 使用工具结果重新生成分析
            enhanced_prompt = f"""基于以下工具查询结果，请提供详细的面试分析：

工具查询结果：
{chr(10).join(tool_results)}

原始面试数据：
简历：{input_data['resume']}
问答：{self._format_qa_conversation(input_data['qa_pairs'])}
音频分析：{' | '.join(input_data['audio_summaries'])}
视频分析：{' | '.join(input_data['video_summaries'])}

请给出综合评估，包括各维度评分、优势劣势、招聘建议等。"""

            enhanced_response = self.llm.invoke([HumanMessage(content=enhanced_prompt)])
            return enhanced_response
        
        return response
    
    def _parse_analysis_result(self, response, input_data: InterviewAnalysisInput) -> AnalysisResult:
        """解析分析结果"""
        try:
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 动态评分系统
            scores = self._calculate_dynamic_scores(input_data, content)
            insights = self._extract_insights_from_content(input_data, content)
            
            return AnalysisResult(
                overall_score=scores["overall"],
                technical_competency=scores["technical"],
                communication_skills=scores["communication"],
                problem_solving=scores["problem_solving"],
                
                strengths=insights["strengths"],
                weaknesses=insights["weaknesses"],
                recommendations=insights["recommendations"],
                
                key_insights=insights["key_insights"],
                behavioral_analysis=self._extract_behavioral_insights(
                    input_data["audio_summaries"], 
                    input_data["video_summaries"]
                ),
                
                detailed_analysis={
                    "analysis_content": content,
                    "timestamp": input_data.get("timestamp", ""),
                    "interview_duration": len(input_data["qa_pairs"]) * 5,
                    "scoring_method": "dynamic_analysis"
                }
            )
            
        except Exception as e:
            print(f"解析分析结果失败: {e}")
            return self._create_fallback_analysis(input_data)
    
    def _create_basic_analysis(self, input_data: InterviewAnalysisInput) -> AnalysisResult:
        """创建基础分析结果（无LLM模式）- 使用动态评分"""
        
        # 使用动态评分系统，传入空的content
        scores = self._calculate_dynamic_scores(input_data, "")
        insights = self._extract_insights_from_content(input_data, "")
        
        return AnalysisResult(
            overall_score=scores["overall"],
            technical_competency=scores["technical"],
            communication_skills=scores["communication"],
            problem_solving=scores["problem_solving"],
            
            strengths=insights["strengths"],
            weaknesses=insights["weaknesses"],
            recommendations=insights["recommendations"],
            
            key_insights=insights["key_insights"],
            behavioral_analysis=self._extract_behavioral_insights(
                input_data["audio_summaries"], 
                input_data["video_summaries"]
            ),
            
            detailed_analysis={
                "mode": "basic_dynamic_analysis",
                "qa_count": len(input_data["qa_pairs"]),
                "audio_segments": len(input_data["audio_summaries"]),
                "video_segments": len(input_data["video_summaries"]),
                "note": "使用动态评分系统（基础模式）",
                "scoring_breakdown": {
                    "qa_score": self._evaluate_qa_quality(input_data.get("qa_pairs", []), input_data.get("resume", "")),
                    "comm_score": self._evaluate_communication(input_data.get("audio_summaries", []), input_data.get("video_summaries", [])),
                    "depth_score": self._evaluate_content_depth(input_data.get("qa_pairs", []), ""),
                    "overall_score": self._evaluate_overall_performance(input_data, "")
                }
            }
        )
    
    def _create_fallback_analysis(self, input_data: InterviewAnalysisInput) -> AnalysisResult:
        """创建备用分析结果"""
        return AnalysisResult(
            overall_score=75.0,
            technical_competency=75.0,
            communication_skills=75.0,
            problem_solving=75.0,
            
            strengths=["基础表现良好"],
            weaknesses=["需要进一步评估"],
            recommendations=["建议安排技术复试"],
            
            key_insights=["面试数据分析中遇到技术问题"],
            behavioral_analysis="分析系统暂时不可用",
            
            detailed_analysis={
                "error": "分析过程遇到问题，使用备用评估",
                "qa_count": len(input_data["qa_pairs"])
            }
        )
    
    def _save_analysis_to_db(self, analysis: AnalysisResult, input_data: InterviewAnalysisInput):
        """保存分析结果到向量数据库"""
        try:
            analysis_text = f"""
面试分析报告：
总体评分：{analysis.overall_score}
技术能力：{analysis.technical_competency}
沟通能力：{analysis.communication_skills}
问题解决能力：{analysis.problem_solving}

优势：{', '.join(analysis.strengths)}
不足：{', '.join(analysis.weaknesses)}
建议：{', '.join(analysis.recommendations)}

关键洞察：{', '.join(analysis.key_insights)}
行为分析：{analysis.behavioral_analysis}
"""
            
            vector_db.add_document(
                text=analysis_text,
                metadata={
                    "source": "interview_analysis",
                    "type": "analysis_report",
                    "overall_score": analysis.overall_score,
                    "candidate_summary": input_data["resume"][:100]
                }
            )
            
            print("✅ 分析结果已保存到知识库")
            
        except Exception as e:
            print(f"保存分析结果失败: {e}")
    
    def _calculate_dynamic_scores(self, input_data: InterviewAnalysisInput, content: str) -> dict:
        """动态计算评分"""
        qa_pairs = input_data.get("qa_pairs", [])
        audio_summaries = input_data.get("audio_summaries", [])
        video_summaries = input_data.get("video_summaries", [])
        resume = input_data.get("resume", "")
        
        # 基础分数
        base_score = 50
        
        # 1. 问答质量评分 (0-30分)
        qa_score = self._evaluate_qa_quality(qa_pairs, resume)
        
        # 2. 沟通表现评分 (0-25分)
        comm_score = self._evaluate_communication(audio_summaries, video_summaries)
        
        # 3. 内容深度评分 (0-20分)
        depth_score = self._evaluate_content_depth(qa_pairs, content)
        
        # 4. 整体表现调整 (0-15分)
        overall_adjustment = self._evaluate_overall_performance(input_data, content)
        
        # 计算各维度分数
        technical_base = base_score + qa_score * 0.8 + depth_score * 0.9
        communication_base = base_score + comm_score * 1.2 + qa_score * 0.3
        problem_solving_base = base_score + qa_score * 0.6 + depth_score * 0.8 + overall_adjustment * 0.6
        
        # 确保分数在合理范围内
        technical = max(30, min(100, technical_base))
        communication = max(30, min(100, communication_base))
        problem_solving = max(30, min(100, problem_solving_base))
        overall = (technical + communication + problem_solving) / 3
        
        return {
            "overall": round(overall, 1),
            "technical": round(technical, 1),
            "communication": round(communication, 1),
            "problem_solving": round(problem_solving, 1)
        }
    
    def _evaluate_qa_quality(self, qa_pairs: list, resume: str) -> float:
        """评估问答质量 (0-30分)"""
        if not qa_pairs:
            return 0
        
        score = 0
        for question, answer in qa_pairs:
            # 回答长度评分
            answer_length = len(answer.strip())
            if answer_length > 50:
                score += 3
            elif answer_length > 20:
                score += 2
            else:
                score += 1
            
            # 关键词匹配评分
            tech_keywords = ["python", "java", "javascript", "算法", "数据库", "框架", "项目", "开发", "技术"]
            keyword_count = sum(1 for keyword in tech_keywords if keyword.lower() in answer.lower())
            score += min(3, keyword_count * 0.5)
            
            # 逻辑性评分（简单判断）
            logical_indicators = ["因为", "所以", "首先", "然后", "最后", "例如", "比如"]
            logic_score = sum(1 for indicator in logical_indicators if indicator in answer)
            score += min(2, logic_score * 0.5)
        
        return min(30, score)
    
    def _evaluate_communication(self, audio_summaries: list, video_summaries: list) -> float:
        """评估沟通表现 (0-25分)"""
        score = 0
        
        # 音频分析评分
        for audio in audio_summaries:
            if "清晰" in audio or "流畅" in audio:
                score += 3
            if "自信" in audio or "稳定" in audio:
                score += 2
            if "紧张" in audio or "不清楚" in audio:
                score -= 2
        
        # 视频分析评分
        for video in video_summaries:
            if "自然" in video or "得当" in video:
                score += 3
            if "专注" in video or "集中" in video:
                score += 2
            if "紧张" in video or "不适" in video:
                score -= 2
        
        return max(0, min(25, score))
    
    def _evaluate_content_depth(self, qa_pairs: list, content: str) -> float:
        """评估内容深度 (0-20分)"""
        if not qa_pairs:
            return 0
        
        score = 0
        total_content = " ".join([answer for _, answer in qa_pairs])
        
        # 技术深度关键词
        advanced_keywords = ["架构", "优化", "性能", "并发", "分布式", "微服务", "设计模式", "算法复杂度"]
        depth_score = sum(2 for keyword in advanced_keywords if keyword in total_content)
        score += min(10, depth_score)
        
        # 问题解决思路
        problem_solving_keywords = ["解决", "处理", "优化", "改进", "分析", "思考", "方案", "策略"]
        solving_score = sum(1 for keyword in problem_solving_keywords if keyword in total_content)
        score += min(5, solving_score)
        
        # 实际经验
        experience_keywords = ["项目", "经验", "实践", "实现", "负责", "开发", "维护", "团队"]
        exp_score = sum(1 for keyword in experience_keywords if keyword in total_content)
        score += min(5, exp_score)
        
        return min(20, score)
    
    def _evaluate_overall_performance(self, input_data: dict, content: str) -> float:
        """评估整体表现 (0-15分)"""
        score = 0
        
        # 完整性评分
        qa_count = len(input_data.get("qa_pairs", []))
        if qa_count >= 3:
            score += 5
        elif qa_count >= 2:
            score += 3
        else:
            score += 1
        
        # 数据质量评分
        audio_count = len(input_data.get("audio_summaries", []))
        video_count = len(input_data.get("video_summaries", []))
        
        if audio_count > 0 and video_count > 0:
            score += 5
        elif audio_count > 0 or video_count > 0:
            score += 3
        
        # 内容长度评分
        total_answer_length = sum(len(answer) for _, answer in input_data.get("qa_pairs", []))
        if total_answer_length > 200:
            score += 5
        elif total_answer_length > 100:
            score += 3
        else:
            score += 1
        
        return min(15, score)
    
    def _extract_insights_from_content(self, input_data: dict, content: str) -> dict:
        """从内容中提取洞察"""
        qa_pairs = input_data.get("qa_pairs", [])
        total_content = " ".join([answer for _, answer in qa_pairs])
        
        # 动态生成优势
        strengths = []
        if "python" in total_content.lower() or "java" in total_content.lower():
            strengths.append("具备编程语言基础")
        if "项目" in total_content or "开发" in total_content:
            strengths.append("有实际项目经验")
        if len(total_content) > 200:
            strengths.append("表达详细充分")
        if "团队" in total_content or "合作" in total_content:
            strengths.append("具备团队协作意识")
        
        # 动态生成劣势
        weaknesses = []
        if len(total_content) < 100:
            weaknesses.append("回答过于简短")
        if not any(keyword in total_content for keyword in ["项目", "经验", "实践"]):
            weaknesses.append("缺少具体实践经验描述")
        if len(qa_pairs) < 2:
            weaknesses.append("面试轮次较少，信息不够充分")
        
        # 动态生成建议
        recommendations = []
        avg_score = (
            self._evaluate_qa_quality(qa_pairs, input_data.get("resume", "")) +
            self._evaluate_communication(input_data.get("audio_summaries", []), input_data.get("video_summaries", []))
        ) / 2
        
        if avg_score > 20:
            recommendations.append("综合表现良好，建议进入下一轮")
        elif avg_score > 15:
            recommendations.append("基础能力达标，可考虑培训后录用")
        else:
            recommendations.append("需要进一步提升技能后再申请")
        
        # 生成关键洞察
        key_insights = []
        if "算法" in total_content or "数据结构" in total_content:
            key_insights.append("候选人具备算法基础")
        if "优化" in total_content or "性能" in total_content:
            key_insights.append("关注系统性能和优化")
        if len(qa_pairs) >= 3:
            key_insights.append("面试参与度高")
        
        return {
            "strengths": strengths if strengths else ["参与面试"],
            "weaknesses": weaknesses if weaknesses else ["需要更多评估"],
            "recommendations": recommendations,
            "key_insights": key_insights if key_insights else ["完成了基础面试流程"]
        }

# 全局分析agent实例
analysis_agent = InterviewAnalysisAgent()