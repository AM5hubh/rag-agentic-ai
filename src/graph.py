from typing import List, TypedDict
import os

# from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END 
from langchain_openai import OpenAIEmbeddings 
from groq import Groq
from langchain_pinecone import PineconeVectorStore

client = Groq()

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float

def build_rag_graph(index_name: str):

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url="https://openrouter.ai/api/v1",
    )

    vectorstore = PineconeVectorStore( 
        index_name=index_name, 
        embedding=embeddings, 
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # llm = ChatOpenAI(model="stealth/space-bunny-alpha", temperature=0.4, base_url='https://openrouter.ai/api/v1')
    

    # Define Nodes
    # def retrieve_node(state: AgentState):
    #     docs = retriever.invoke(state["question"])
    #     context_texts = [d.page_content for d in docs]
    #     return {"context": context_texts}

    # def retrieve_node(state: AgentState):
    #     docs_with_scores = vectorstore.similarity_search_with_score(
    #         state["question"],
    #         k=4
    #     )

    #     context_texts = []
    #     scores = []

    #     for doc, score in docs_with_scores:
    #         context_texts.append(doc.page_content)
    #         scores.append(score)

    #         print("Score:", score)
    #         print("Content:", doc.page_content[:200])
    #         print("---")

    #     # Highest/average score depends on your Pinecone metric.
    #     avg_score = sum(scores) / len(scores) if scores else 0.0

    #     return {
    #         "context": context_texts,
    #         "score": avg_score,
    #     }
    # 
    
    def retrieve_node(state: AgentState):
        question = state["question"]

        # Retrieve the top 10 candidates from Pinecone
        docs_with_scores = vectorstore.similarity_search_with_score(
            question,
            k=10,
        )

        # Higher cosine similarity = more relevant
        ranked_docs = sorted(
            docs_with_scores,
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # Remove duplicate chunks while preserving relevance order
        unique_docs = []
        seen_content = set()

        for doc, score in ranked_docs:
            normalized_content = " ".join(doc.page_content.split()).casefold()

            if normalized_content in seen_content:
                continue

            seen_content.add(normalized_content)
            unique_docs.append((doc, float(score)))

            if len(unique_docs) == 4:
                break

        context = []
        scores = []

        for rank, (doc, score) in enumerate(unique_docs, start=1):
            context.append(doc.page_content)
            scores.append(score)

            print(f"\nRank: {rank}")
            print(f"Cosine Similarity: {score:.4f}")
            print(f"Content: {doc.page_content[:300]}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 60)

        return {
            "context": context,
            "score": max(scores) if scores else 0.0,
        }


    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
        If the context does not contain enough info, state 'I cannot answer based on the provided document.'

        Context:
        {context_str}

        Question: {state['question']}"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": prompt}
            ],temperature=0.4,
        )

        # Simple score heuristic based on retrieved context availability
        # confidence = 0.95 if len(state["context"]) > 0 else 0.0

        return {"answer": response.choices[0].message.content}

    # Build Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()