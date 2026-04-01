import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant


load_dotenv()

COLLECTION_NAME = "youtube_knowledge"


llm = ChatOpenAI(model="gpt-4o-mini")


# -------------------------
# Connect Qdrant
# -------------------------

client = QdrantClient(":memory:")

vectordb = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=OpenAIEmbeddings()
)


# -------------------------
# State
# -------------------------

class AgentState(TypedDict):

    question: str
    context: str
    answer: str


# -------------------------
# Retriever Node
# -------------------------

def retriever_agent(state: AgentState):

    retriever = vectordb.as_retriever()

    docs = retriever.invoke(state["question"])

    context = "\n".join([d.page_content for d in docs])

    return {"context": context}


# -------------------------
# Research Agent
# -------------------------

def research_agent(state: AgentState):

    prompt = f"""
You are a YouTube research assistant.

Use the context from the video transcript to answer the question.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


# -------------------------
# Graph
# -------------------------

builder = StateGraph(AgentState)

builder.add_node("retriever", retriever_agent)
builder.add_node("researcher", research_agent)

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "researcher")
builder.add_edge("researcher", END)

graph = builder.compile()


# -------------------------
# Autonomous Loop
# -------------------------

def run_agent():

    print("\nAutonomous YouTube Research Agent")
    print("Type 'exit' or 'quit' to stop\n")

    while True:

        question = input("\nAsk Question: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting agent...")
            break

        result = graph.invoke({
            "question": question
        })

        print("\nAnswer:\n", result["answer"])


if __name__ == "__main__":

    run_agent()