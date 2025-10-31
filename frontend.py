import streamlit as st
import requests
from datetime import datetime

# Backend URL
BACKEND_URL = "http://localhost:8000/chat"

# Streamlit page config
st.set_page_config(page_title="RTRWH Chatbot", page_icon="💧", layout="wide")

st.title("💧 RTRWH & Artificial Recharge Chatbot")
st.write("Ask me anything about **Rooftop Rainwater Harvesting** and **Artificial Recharge**!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {msg['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {msg['content']}")
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

# User input box (chat-style)
user_input = st.chat_input("Type your question here...")

# If user sends a message
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    # Send the message to backend
    payload = {"message": user_input}

    try:
        resp = requests.post(BACKEND_URL, json=payload)

        # Validate response
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                st.error("⚠️ The server returned an invalid JSON response.")
                st.stop()

            # Extract response details
            answer = data.get("answer", "I'm not sure how to answer that.")
            sources = data.get("sources", [])

            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {answer}")
                if sources:
                    st.caption("Sources: " + ", ".join(sources))

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        else:
            st.error(f"Server error {resp.status_code}: {resp.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
