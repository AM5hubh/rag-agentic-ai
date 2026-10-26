import requests
import streamlit as st

# -------------------- Page Configuration --------------------

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

API_URL = "http://localhost:8000/chat"

st.title("🤖 Agentic AI RAG Chatbot")
st.caption("Ask questions about the Agentic AI eBook.")

# -------------------- Initialize Chat History --------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm your Agentic AI assistant. "
                "Ask me anything about the Agentic AI eBook."
            ),
        }
    ]

# -------------------- Display Chat History --------------------

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display retrieval information for assistant responses
        if message["role"] == "assistant" and "retrieved_context" in message:

            with st.expander("📚 Retrieved Context"):
                if message["retrieved_context"]:
                    for i, chunk in enumerate(
                        message["retrieved_context"], start=1
                    ):
                        st.markdown(f"**Chunk {i}**")
                        st.write(chunk)
                        st.divider()
                else:
                    st.info("No relevant context was retrieved.")

            if message.get("retrieval_similarity_score") is not None:
                score = message["retrieval_similarity_score"]
                st.caption(
                    f"Retrieval similarity: {score * 100:.2f}%"
                )

            elif message.get("confidence_score") is not None:
                score = message["confidence_score"]
                st.caption(
                    f"Confidence score: {score * 100:.2f}%"
                )

# -------------------- Chat Input --------------------

query = st.chat_input("Ask something about the Agentic AI eBook...")

if query:
    # Display the user's message immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching the eBook and generating an answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=120,
                )

                response.raise_for_status()
                data = response.json()

                answer = data.get(
                    "final_answer",
                    "The API did not return an answer.",
                )

                retrieved_context = data.get("retrieved_context", [])

                # Support either API field name
                score = data.get(
                    "retrieval_similarity_score",
                    data.get("confidence_score"),
                )

                st.markdown(answer)

                with st.expander("📚 Retrieved Context"):
                    if retrieved_context:
                        for i, chunk in enumerate(
                            retrieved_context, start=1
                        ):
                            st.markdown(f"**Chunk {i}**")
                            st.write(chunk)
                            st.divider()
                    else:
                        st.info("No relevant context was retrieved.")

                if score is not None:
                    st.caption(
                        f"Retrieval similarity: {score * 100:.2f}%"
                    )

                # Save the response for future reruns
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "retrieved_context": retrieved_context,
                        "retrieval_similarity_score": score,
                    }
                )

            except requests.exceptions.Timeout:
                error_message = (
                    "The request timed out. The model may be taking "
                    "too long to respond. Please try again."
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except requests.exceptions.ConnectionError:
                error_message = (
                    "Could not connect to the RAG API. "
                    "Make sure your FastAPI server is running."
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except requests.exceptions.HTTPError as error:
                error_message = (
                    f"The API returned an HTTP error: {error}"
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            except (ValueError, KeyError) as error:
                error_message = (
                    f"Could not process the API response: {error}"
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )