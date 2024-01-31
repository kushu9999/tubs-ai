import os
import logging
from utils import BedrockLLM
from tools import total_tools
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Configure logging to a file
logging.basicConfig(filename='logs/error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


# inference agent
def inference_agent(query:str):

    llm = BedrockLLM.get_bedrock_llm()
    # llm = BedrockLLM.get_bedrock_llm(model_id="anthropic.claude-v2:1")

    # custom prompt for our agent.
    custom_prefix = f""" \n\n'Human:' You are a Docs Expert AI chatbot called Doxpert. \n
    Stricky don't use knowledge or personal thoughts and opition, just using tools and get answer from that tools.
    You have to answer Human questions related to various agreements, contracts, acts and templates using various available tools below. \n
    'action_input' for the tools are simple questions related to Human query. \n
    Always return answer from Observation, Strictly don't use your knowledge in final Thought. \n
    If you get '_______' say as per decided in an agreement or a contract don't put your data. \n
    You can create an agreement or a contract using tools on Human's request. \n
    Below are list of tools you can use: \n
    """

    custom_suffix = """
    \n\n'Assistant:' You're limited to a maximum of 3 iterations, not unlimited.
    Always embody the persona of Doxpert, the Docs Expert AI.
    Strickly don't mentioned using which tool or how you get the results,
    just answer final answer as assistant.
    Don't forget that you do not have to use your knowledge and data.
    Below are our conversation.
    """

    """
    This memory if for INTERNAL storage of chat history.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, ai_prefix='Assistant')


    """
    Agent executor for agent.
    """
    chat_message_int = MessagesPlaceholder(variable_name="chat_history")
    conversational_agent = initialize_agent(
        tools=total_tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        max_iterations=3,
        verbose=True,
        agent_kwargs={
            "prefix": custom_prefix,
            "suffix": custom_suffix,
            "memory_prompts": [chat_message_int],
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
            "stop": ["\nObservation:"],
        },
    )

    try:
        # executing agent
        response = conversational_agent(query)
        # print("final result ", response['output'])
        return response['output']

    except Exception as e:
        # Handling exceptions
        print(f"An unexpected error occurred while executing the agent: {e}")
        logging.error(f"An unexpected error occurred while executing the agent: {e}", exc_info=True)
        return "ERROR An unexpected error occurred while executing the agent: {e}"


if __name__ == "__main__":
    while True:
        query = input("Please enter your question: ")
        if query.lower() == "bye":
            break

        result = inference_agent(query)

        print("Answer:",result)