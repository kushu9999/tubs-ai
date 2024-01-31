from langchain.agents import Tool
from utils import BedrockLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings

"""
Vector Store
"""
def base_model(vectorstore_path):
    vectorstore_path = vectorstore_path
    # AWS Bedrock embeddings
    bedrock_runtime_client = BedrockLLM.get_bedrock_client()
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_runtime_client)

    vector_store = FAISS.load_local(vectorstore_path, embeddings)
    llm = BedrockLLM.get_bedrock_llm()

    # Create a question answering system based on information retrieval using the RetrievalQA class, which takes as input a neural language model, a chain type and a retriever (an object that allows you to retrieve the most relevant chunks of text for a query)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 6}))

    return qa


def rental_agreements(query:str):
    qa = base_model(vectorstore_path = "models/rental-models/faiss_index")
    result = qa(query)
    return result


rental_agreement_tool = Tool.from_function(
    name="search related to rental agreement",
    func=rental_agreements,
    description="This tool is useful to get answers related to various rental agreements. use this tool if you get keywords like 'rental', 'house rent' etc",
)

def car_lease_agreements(query:str):
    qa = base_model(vectorstore_path = "models/car-lease-models/faiss_index")
    result = qa(query)
    return result


car_lease_agreement_tool = Tool.from_function(
    name="search related to car lease agreement",
    func=car_lease_agreements,
    description="This tool is useful to get answers related to various car lease agreements. use this tool if you get keywords like 'car lease', 'car rent' etc",
)

def business_partnership_agreements(query:str):
    qa = base_model(vectorstore_path = "models/business-partnership-models/faiss_index")
    result = qa(query)
    return result


business_partnership_agreements_tool = Tool.from_function(
    name="search related to business partnership agreement",
    func=business_partnership_agreements,
    description="This tool is useful to get answers related to various business partnership agreements. use this tool if you get keywords like 'business partnership', 'business partner' etc",
)

def contracts(query:str):
    qa = base_model(vectorstore_path = "models/contract-act-models/faiss_index")
    result = qa(query)
    return result


contract_act_tool = Tool.from_function(
    name="search related to contracts",
    func=contracts,
    description="This tool is useful to get answers related to various contracts. use this tool if you get keywords like 'scotland contract act', 'contract act 1990', 'contract for services', 'contract addendum' etc",
)

def immigration(query:str):
    qa = base_model(vectorstore_path = "models/immigration/faiss_index")
    result = qa(query)
    return result


immigration_tool = Tool.from_function(
    name="search related to new foundland immigration",
    func=immigration,
    description="This tool is useful to get answers related to various queries related to new foundland immigration. use this tool if you get keywords like 'immigration', 'new foundland', 'newfoundland', 'new foundland immigration' etc",
)

total_tools = [rental_agreement_tool, car_lease_agreement_tool, business_partnership_agreements_tool, immigration_tool]
