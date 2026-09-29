from typing import List, TypedDict
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END 
from langchain_openai import ChatOpenAI 
from langchain_pinecone import PineconeVectorStore

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore( 
        index_name=index_name, 
        embedding=embeddings, 
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="stealth/space-bunny-alpha", temperature=0,base_url='https://openrouter.ai/api/v1')

    # Define Nodes
    # def retrieve_node(state: AgentState):
    #     docs = retriever.invoke(state["question"])
    #     context_texts = [d.page_content for d in docs]
    #     return {"context": context_texts}

    def retrieve_node(state: AgentState):
        docs_with_scores = vectorstore.similarity_search_with_score(
            state["question"],
            k=5
        )

        context_texts = []
        scores = []

        for doc, score in docs_with_scores:
            context_texts.append(doc.page_content)
            scores.append(score)

            print("Score:", score)
            print("Content:", doc.page_content[:200])
            print("---")

        # Highest/average score depends on your Pinecone metric.
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "context": context_texts,
            "score": avg_score,
        }

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
        If the context does not contain enough info, state 'I cannot answer based on the provided document.'

        Context:
        {context_str}

        Question: {state['question']}"""

        response = llm.invoke(prompt)

        # Simple score heuristic based on retrieved context availability
        # confidence = 0.95 if len(state["context"]) > 0 else 0.0

        return {"answer": response.content}

    # Build Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()