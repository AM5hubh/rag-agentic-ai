import requests 
import streamlit as st

st.title("Agentic AI RAG Chatbot")

query = st.chat_input( "Ask something about the Agentic AI eBook" )

if query: 
    response = requests.post( 
        "https://rag-agentic-ai-bni1.onrender.com/chat", 
        json={ "query": query }, 
        timeout=60, 
    ) 
    data = response.json() 
    st.subheader("Answer")
    st.write( data["final_answer"] ) 
    st.subheader("Confidence Score") 
    st.write( data["confidence_score"] ) 
    st.subheader("Retrieved Context") 
    for index, chunk in enumerate( data["retrieved_context"], start=1, ): 
        with st.expander( f"Context Chunk {index}" ): 
            st.write(chunk)