from typing import TypedDict, List, Annotated, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class InterviewDecision(BaseModel):
    next_question: Optional[str] = Field(None, description="下一个面试问题")
    should_round: bool = Field(False, description="是否继续问答")

def create_question_agent():
    """创建问题生成agent"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名资深中文面试官。

基于候选人简历和对话历史：
1. 如果上一轮回答有可深挖之处，基于该回答追问1个问题
2. 否则，从简历中选一处尚未被问到的要点，提出1个全新问题  
3. 若已没有可问的问题，设置should_round为False表示结束

候选人简历：{resume}"""),
        ("placeholder", "{messages}")
    ])
    
    llm = ChatOpenAI(model='gpt-4o')
    structured_llm = llm.with_structured_output(InterviewDecision)
    return prompt | structured_llm

