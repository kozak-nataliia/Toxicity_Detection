# toxicity-agent-gradio/app/agent.py
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

from model import ToxicModel
from rag import load_retriever

load_dotenv()
OPENAI = os.getenv("OPENAI_API_KEY")

if OPENAI:
    from langchain_openai import ChatOpenAI
    def make_llm():
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    def make_llm():
        return None

@dataclass
class Pipeline:
    llm: Optional[object]
    toxic: ToxicModel
    agent: Optional[AgentExecutor]  # updated type hint
    rag_tool_name: str = "student_profile_rag"
    toxic_tool_name: str = "toxicity_classifier"

def make_pipeline() -> Pipeline:
    model = ToxicModel()

    @tool("toxicity_classifier")
    def toxicity_classifier(text: str) -> dict:
        """Classify text toxicity. Returns {'toxic_prob': float, 'label': int}."""
        p, y = model.predict(text or "")
        return {"toxic_prob": round(float(p), 4), "label": int(y)}

    retriever = load_retriever()
    rag_tool = create_retriever_tool(
        retriever,
        name="student_profile_rag",
        description="Answer questions about the student using the indexed student_profile.md",
    )

    llm = make_llm()
    if llm:
        tools = [toxicity_classifier, rag_tool]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You have two tools:\n"
                 "1) toxicity_classifier: detect toxicity in text.\n"
                 "2) student_profile_rag: answer questions about the student using retrieved context.\n"
                 "Pick the right tool. When using RAG, ground answers in the retrieved text."
                 ),
                ("human", "{input}"),
                # REQUIRED for tool agents:
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        _agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        agent_exec = AgentExecutor(agent=_agent, tools=tools, verbose=False)
    else:
        agent_exec = None

    return Pipeline(llm=llm, toxic=model, agent=agent_exec)
