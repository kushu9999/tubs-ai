import os
import boto3
from langchain.llms.bedrock import Bedrock
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')


class BedrockLLM:

    @staticmethod
    def get_bedrock_client():

        bedrock_runtime_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )

        return bedrock_runtime_client

    @staticmethod
    def get_bedrock_llm(model_id:str = "anthropic.claude-instant-v1", max_tokens_to_sample:int = 10000, temperature:float = 0.0, top_k:int = 250, top_p:int = 1):

        params = {
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }

        bedrock_llm = Bedrock(
            model_id=model_id,
            client=BedrockLLM.get_bedrock_client(),
            model_kwargs=params,
        )

        return bedrock_llm
