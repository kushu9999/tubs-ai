import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from llm import tubs_xray

llm = ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview")
llm2 = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106")

# Constants
LOGO_PATH = "./images/small-logo.png"
USER_AVATAR_PATH = "./images/user-white.jpeg"

# Function to handle user input
def handle_user_input(query):
    user_avatar = USER_AVATAR_PATH
    message = st.chat_message("user", avatar=user_avatar)
    message.write(query)
    st.session_state.messages.append({"role": "user", "content": query, "avatar": user_avatar})

    with st.spinner('Getting results ....'):
        try:
            result = tubs_xray(query=query)
            tubs_avatar = LOGO_PATH
            message = st.chat_message("assistant", avatar=tubs_avatar)
            message.write(result.content)
            print(result.content)
            st.session_state.messages.append({"role": "assistant", "content": result.content, "avatar": tubs_avatar})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# Streamlit configuration
st.set_page_config(page_title=f"MOXI")
st.header("MOXI 👩🏻‍💼")

# Initialize session_state.messages with avatar information
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MOXI AI assistant.", "avatar": LOGO_PATH}
    ]

# Display existing chat history with avatars
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# Get user input
if query := st.chat_input("Say something"):
    handle_user_input(query)
