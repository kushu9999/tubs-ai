import streamlit as st
from datetime import datetime
import time
import threading
from agent import inference_agent

st.set_page_config(page_title=f"Docs Expert 📄📜")
st.header("Docs Expert AI 📄📜")

# Initialize session_state.messages with avatar information
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Docs Expert AI assistant.", "avatar": "./images/avatar-logo.png"}
    ]

# Display existing chat history with avatars
for message in st.session_state.messages:

    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

if query := st.chat_input("Say something"):
    user_avatar = "./images/user.png"  # Set user avatar

    st.session_state.messages.append({"role": "user", "content": query, "avatar": user_avatar})
    message = st.chat_message("user", avatar=user_avatar)
    message.write(query)

    with st.spinner('Getting results ....'):
        try:
            result = inference_agent(query)
            lawyer_avatar = "./images/avatar-logo.png"
            message = st.chat_message("assistant", avatar=lawyer_avatar)
            message.write(result)

            st.session_state.messages.append({"role": "assistant", "content": result, "avatar": lawyer_avatar})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
