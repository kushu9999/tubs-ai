import streamlit as st
import requests
from agent import inference_agent

USER_AVATAR_PATH = "./images/user-white.jpeg"
LOGO_PATH = "./images/small-logo.png"

# Setting title
st.set_page_config(page_title=f"Docs Expert 📄📜")
st.header("Docs Expert AI 📄📜")


# Initialize session_state.messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if query := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": query, "avatar": USER_AVATAR_PATH})
    message = st.chat_message("user", avatar=USER_AVATAR_PATH)
    message.write(query)

    with st.spinner('Getting results ....'):
        try:
            result = inference_agent(query)
            message = st.chat_message("assistant", avatar=LOGO_PATH)
            message.write(result)

            st.session_state.messages.append({"role": "assistant", "content": result, "avatar": LOGO_PATH})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
