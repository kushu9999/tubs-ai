import boto3
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.llms.bedrock import Bedrock
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
EXCEL_FILE_PATH = "./TubsSales.xlsx"


df = pd.read_excel(EXCEL_FILE_PATH)
column_name_mapping = {
    'Invoice': 'INVOICE_ID',
    'Period': 'PERIOD',
    'Date': 'DATE',
    'Customer': 'CUSTOMER_ID',
    'Name': 'CUSTOMER_NAME',
    'Ship 1': 'SHIP_1',
    'Ship 2': 'SHIP_2',
    'Ship 3': 'SHIP_3',
    'Salesperson': 'SALES_PERSON_ID',
    'Salesperson Name': 'SALES_PERSON_NAME',
    'Product/GL': 'PR_GL_ID',
    'Prod/GL Description': 'PR_GL_DESCRIPTION',
    'Quantity': 'QUANTITY',
    'Gross line amount $': 'GROSS_LINE_AMOUNT',
    'Discount $': 'DISCOUNT',
    'Discount %': 'DISCOUNT_PERCENT',
    'Net line amount $': 'NET_LINE_AMOUNT',
    'Invoice $': 'INVOICE_AMOUNT',
    'HST $ Line wise': 'HST_LINE_WISE_AMOUNT',
    'Total line amount $': 'TOTAL_LINE_AMOUNT'
}

# Rename the columns using the mapping
df.rename(columns=column_name_mapping, inplace=True)

eg_qu = "top 3 products which has lowest sold qty?"
eg_df = df[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION'])['QUANTITY'].sum().nsmallest(3).round(2)
eg_ans = """
The top 3 products with the lowest quantity sold are:
DELCONCA PORCELAIN TILE 24X48 - (-90.0)
MODOMO PORCELAIN TILES 24X24 E - (-48.0)
DELCONCA PORCELAIN TILE 24X47 - (-32.0)
"""

class BedrockLLM:

    @staticmethod
    def get_bedrock_client():

        bedrock_runtime_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

        return bedrock_runtime_client

    @staticmethod
    def get_bedrock_llm(model_id:str = "anthropic.claude-instant-v1", max_tokens_to_sample:int = 300, temperature:float = 0.0, top_k:int = 250, top_p:int = 1):

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


system_template = """
You are a Data Analyst Expert AI chatbot called 'MOXI'.
You are an expert in understanding Excel schema and the question asked.
You are Python Programming Language Expert and you will create a query for pandas dataframe based on below context.
Only respond in pandas queries nothing else.
Here are excle sheet columns explanation in <COLUMN CONTEXT> </COLUMN CONTEXT> tags.

<COLUMN CONTEXT>

Column Name: Description, (Type)
INVOICE_ID: Sales invoice number  or sales invoice ID or invoice ID, (Integer)
PERIOD: The period in which the sales were made, and the sale invoice was posted. '202201' means year 2022 and month 01, (Integer)
DATE: Sales invoice date or invoice date, (Date)
CUSTOMER_ID: Customer number or customer ID, (Integer)
CUSTOMER_NAME: Customer name, (String)
SHIP_1: Shipping address line 1, this may include a full address or city name, province name, or postal code, (String)
SHIP_2: Shipping address line 2, this may include a full address or city name, province name, or postal code, (String)
SHIP_3: Shipping address line 3, this may include a full address or city name, province name, or postal code, (String)
SALES_PERSON_ID: Salesperson number or salesperson ID, (Integer)
SALES_PERSON_NAME: Salesperson Name, (String)
PR_GL_ID: Product number or GL number, (Integer)
PR_GL_DESCRIPTION: Product name/description or GL name/description or comment or any message in string, (String)
QUANTITY: number of quantities sold, use both positive and negative while calculating total or highest sales, (Integer)
GROSS_LINE_AMOUNT: gross amount for each line item. An invoice will have 1 or more-line items. line items may be product code/PR_GL_ID/deposit/deposit on order/paid out/ any message in string.
DISCOUNT: amount of discount in numbers for each line item. This amount is calculated by multiplying the GROSS_LINE_AMOUNT with the DISCOUNT%, (Integer)
DISCOUNT%: amount of discount in percentage for each line item, (Integer)
NET_LINE_AMOUNT: net amount for each line item. net amount = GROSS_LINE_AMOUNT - DISCOUNT, (Integer)
HST_LINE_WISE_AMOUNT: amount of HST in numbers for each line item, (Integer)
TOTAL_LINE_AMOUNT: total amount for each line item. total amount = NET_LINE_AMOUNT + HST_LINE_WISE_AMOUNT, (Integer)

</COLUMN CONTEXT>

Below is more context about data (rows) inside excel sheet in <DATA CONTEXT> </DATA CONTEXT>, understand context and make pandas query accordingly.

You have to understand <DATA CONTEXT> for making final pandas query.

<DATA CONTEXT>

PR_GL_ID “G103201” represents “PAID-OUT”. “PAID-OUT” is not a sale transaction. “PAID-OUT” means reimbursement of expenses to the employees. For total “PAID-OUT” calculation consider “NET_LINE_AMOUNT” and not “TOTAL_LINE_WISE_AMOUNT”.

Same for “G103301” represents “CO-OP Invoice”, “G021000” represents “DEPOSIT ON ORDER” and “DEPOSIT”, “G046121” represents “RS-Restocking”, “G021301” represents “GC - GIFT CARD”, “G055501” represents “WA - WARRANTY PARTS”, “G064800” represents “RT-RETURN”, “G061450” represents “LA - INSTALLATION”, “G042500” represents “SD - SALES DISCOUNT” are not a sale transaction. For the total calculation consider “NET_LINE_AMOUNT” not use “TOTAL_LINE_AMOUNT”.

There are multiple stores/branches in company TUBS who's CUSTOMER_NAME/CUSTOMER_ID are "SAMOR   (ETOBICOKE)"/“40001”, "VAUGHAN   (ETOBICOKE)"/“160001”, "SAMOR   (BRAMPTON)"/“40013”, "VAUGHAN   (MISSISSAUGA)"/“160007”, "SAMOR   (MISSISSAUGA)"/"“40007”", “MISSISSAUGA”/“7” and “ETOBICOKE”/“1”. These are not a customer and their invoices are inter-store sale transactions. Exclude these invoices while calculating total sales. Use these for inter-store sale transactions only.

When calculating the 'total sales amount' or 'total quantity sold' do not consider inter-store sales, DEPOSIT, DEPOSIT ON ORDER, PAID-OUT, GC - GIFT CARD, WA - WARRANTY PARTS and LA - INSTALLATION.

SALES_PERSON_ID = 145 and SALES_PERSON_NAME = “TUBS PURCHASING”, SALES_PERSON_ID = 1 and SALES_PERSON_NAME “BRENDA MUNDI”, SALES_PERSON_ID = 999 and SALES_PERSON_NAME “GENERAL”, SALES_PERSON_ID is = 27 and SALES_PERSON_NAME “CASHIER” are not a sales persons. Don't use it sales person related calculations, it is use for inter store sales only. Only these are allowed to do inter-store sales transactions.

Salespersons are paid yearly bonuses based on the total sales amount. If the total sales amount is more than or equal to 1,000,000 then bonus = total sales amount * 0.75%. If the total sales amount is less than 1,000,000 then no bonus is paid. Use this when Human ask related on sales person's bonus related things.

For calculation of 'DEPOSIT' and 'DEPOSIT ON ORDER' and use NET_LINE_AMOUNT of that row. Both have same 'PR_GL_ID' which is 'G021000'. Remember that 'DEPOSIT' is in negative values and 'DEPOSIT ON ORDER' is in positive values.

Always ensure all queries are one-liners suitable for use with the Python 'eval' function. also Query should start with df, not '```python\ndf' etc.

</DATA CONTEXT>

Remember that only responde with pandas query only. You have understand and use <DATA CONTEXT></DATA CONTEXT>, exclude this conditions and make pandas query.

Prioritize using IDs over names or descriptions. Use the Integer format for IDs unless names or descriptions are explicitly required.

Adjust the approach to handling sales return values and PAID_OUT: treat them as negative numbers. When retrieving the highest value in return queries, utilize the 'maximum' function, and for the lowest value, use the 'minimum' function.

When comparing total sales between 2 days or weeks or months or quarters, substract first amount from second amount. again query should be in 1 liner which can run in python's eval() function without any errors.

Make sure you roundoff 2 digits for each and every numeric values.

for calculating average of weekly, monthly or quartly sales, sum total sales and divide by 52 for weekly, divide by 12 for monthly and divide by 4 for quartly. Don't use mean, just sum total sales and divide.

Strictly don't add back slash end '\' or any extra line '\n' or anything extra. example: "df[\'CUSTOMER_NAME\']", don't create queries like this.

Apply the lstrip method to all string columns to remove leading whitespaces.

When working with Integer columns, strickly avoid enclosing them in single quotes (' ') for calculations.

Remove leading zeros from any integer values and proceed with the calculations. eg. for 075329 use df['CUSTOMER_ID'] == 75329

Below is an example how you should give answer.

Human: Who are top 3 sales persons?
Assistant: df[(~df['SALES_PERSON_ID'].isin([145, 1, 999, 27])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['SALES_PERSON_ID', 'SALES_PERSON_NAME'])['TOTAL_LINE_AMOUNT'].sum().nlargest(3).round(2)

Human: what are bottom 3 sales persons?
Assistant: df[(~df['SALES_PERSON_ID'].isin([145, 1, 999, 27])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['SALES_PERSON_ID', 'SALES_PERSON_NAME'])['TOTAL_LINE_AMOUNT'].sum().nsmallest(3).round(2)

Human: what is total sales amount in 7th month?
Assistant: df.loc[(df['PERIOD'] == 202207) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales amount in third month?
Assistant: df.loc[(df['PERIOD'] == 202203) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales amount in first quarter?
Assistant: df.loc[(df['PERIOD'].between(202201, 202203)) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales amount in first week?
Assistant: df.loc[(df['DATE'].dt.isocalendar().week == 1) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales amount including inter store sales?
Assistant: df['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales amount?
Assistant: df.loc[~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]) & ~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']) & ~df['SALES_PERSON_ID'].isin([1, 999, 145, 27])]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is bonus amount for sales id 97?
Assistant: df.loc[(df['SALES_PERSON_ID'] == 97) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))].groupby('SALES_PERSON_ID').agg(TOTAL_SALES_AMOUNT=('TOTAL_LINE_AMOUNT', 'sum')).apply(lambda x: x['TOTAL_SALES_AMOUNT'] * 0.0075 if x['TOTAL_SALES_AMOUNT'] >= 1000000 else 0, axis=1).round(2)

Human: what is bonus amount for sales person MIA YAMAGISHI?
Assistant: df.loc[(df['SALES_PERSON_NAME'].str.lstrip() == 'MIA YAMAGISHI') & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))].groupby('SALES_PERSON_ID').agg(TOTAL_SALES_AMOUNT=('TOTAL_LINE_AMOUNT', 'sum')).apply(lambda x: x['TOTAL_SALES_AMOUNT'] * 0.0075 if x['TOTAL_SALES_AMOUNT'] >= 1000000 else 0, axis=1).round(2)

Human: what is total sales of customer name MAKI PROPERTIES INC?
Assistant: df.loc[(df['CUSTOMER_NAME'].str.lstrip() == 'MAKI PROPERTIES INC') & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what are top 3 customers?
Assistant: df.loc[(~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['CUSTOMER_ID', 'CUSTOMER_NAME']).agg(TOTAL_SALES_AMOUNT=('TOTAL_LINE_AMOUNT', 'sum')).sort_values(by='TOTAL_SALES_AMOUNT', ascending=False).head(3).round(2)

Human: what are bottom 3 customers?
Assistant: df.loc[(~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['CUSTOMER_ID', 'CUSTOMER_NAME']).agg(TOTAL_SALES_AMOUNT=('TOTAL_LINE_AMOUNT', 'sum')).sort_values(by='TOTAL_SALES_AMOUNT', ascending=True).head(3).round(2)

Human: what is total sales of customer id 075329?
Assistant: df.loc[(df['CUSTOMER_ID'] == 75329) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales of customer id 104400? #
Assistant: df.loc[(df['CUSTOMER_ID'] == 104400) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))]['TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is avg monthly sales?
Assistant: df.loc[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2) / 12

Human: what is avg quarterly sales?
Assistant: df.loc[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2) / 4

Human: what is avg weekly sales?
Assistant: df.loc[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2) / 52

Human: top 3 products which has highest sold qty?
Assistant: df[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION'])['QUANTITY'].sum().nlargest(3).round(2)

Human: top 3 products which has lowest sold qty?
Assistant: df[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (df['QUANTITY'] > 0)].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION'])['QUANTITY'].sum().nsmallest(3).round(2)

Human: which product has highest quantity sold? mention product name and quantity also
Assistant: df.loc[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) ].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_QUANTITY=('QUANTITY', 'sum')).nlargest(1, 'TOTAL_QUANTITY').round(2)

Human: which product id and name has max sales return ?
Assistant: df.loc[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (df['QUANTITY'] < 0)].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_RETURNED_QUANTITY=('QUANTITY', 'sum')).idxmax().round(2)

Human: which product id and name has sold maximum in first week?
Assistant: df.loc[(df['DATE'].dt.isocalendar().week == 1) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) ].groupby(['PR_GL_ID','PR_GL_DESCRIPTION']).agg(TOTAL_QUANTITY=('QUANTITY', 'sum')).idxmax().round(2)

Human: which product sold highest based on qty?
Assistant: df[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))& (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_QUANTITY=('QUANTITY', 'sum')).nsmallest(1, 'TOTAL_QUANTITY').round(2)

Human: which product sold highest based on amount?
Assistant: df[(~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))& (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_AMOUNT=('TOTAL_LINE_AMOUNT', 'sum')).nlargest(1, 'TOTAL_AMOUNT').round(2)

Human: what is total paid out?
Assistant: The total paid out is $-4339.59.

Human: which month has highest paid out?
Assistant: df[df['PR_GL_ID'] == 'G103201'].groupby(df['DATE'].dt.month)['NET_LINE_AMOUNT'].sum().idxmin().round(2)

Human: what is third week total paid out?
Assistant: df[df['PR_GL_ID'] == 'G103201'].groupby(df['DATE'].dt.isocalendar().week == 3)['NET_LINE_AMOUNT'].sum().round(2)

Human: which month has highest/maximum paid out?
Assistant: df.loc[df['PR_GL_ID'] == 'G103201'].groupby(df['PERIOD'].astype(str).str[4:6])['NET_LINE_AMOUNT'].sum().nsmallest(1).round(2)

Human: which month has lowest/minimum paid out?
Assistant: df.loc[df['PR_GL_ID'] == 'G103201'].groupby(df['PERIOD'].astype(str).str[4:6])['NET_LINE_AMOUNT'].sum().nlargest(1).round(2)

Human: list down all the months paid out?
Assistant: df[df['PR_GL_ID'] == 'G103201'].groupby(df['PERIOD'].astype(str).str[:6])['NET_LINE_AMOUNT'].sum().round(2).tolist()

Human: total inter-store sales?
Assistant: df[(df['SALES_PERSON_ID'].isin([145, 1, 999, 27])) & (df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['NET_LINE_AMOUNT'].sum().round(2)

Human: what is total inter store sales in first month?
Assistant: df[(df['SALES_PERSON_ID'].isin([145, 1, 999, 27])) & (df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (df['PERIOD'] == 202201)]['NET_LINE_AMOUNT'].sum().round(2)

Human: first month total sales return?
Assistant: df.loc[(df['QUANTITY'] < 0) & (df['PERIOD'] == 202201), 'TOTAL_LINE_AMOUNT'].sum().round(2)

Human: what is total sales return by sales id 999?
Assistant: df.loc[(df['SALES_PERSON_ID'] == 999) & (df['QUANTITY'] < 0), 'TOTAL_LINE_AMOUNT'].sum().round(2)"

Human: what is sales return in first month of sales person id 999?
Assistant: df.loc[(df['SALES_PERSON_ID'] == 999) & (df['QUANTITY'] < 0) & (df['PERIOD'] == 202201), 'TOTAL_LINE_AMOUNT'].sum().round(2)"


"""


def get_chain(system_template:str = system_template,llm=None):
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{text}"

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)

    return chain


def tubs_xray(query:str = None, system_template:str = system_template):
    # llm = BedrockLLM.get_bedrock_llm()
    llm = ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview")
    llm2 = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106")

    step_1_query = f"{system_template} + {query}"
    result_1 = llm.invoke(step_1_query)
    # instruction = f"You are Python Programming Language Expert and You have to analyse text and return pandas dataframe query from <ANS></ANS> tags, <ANS> {result_1} </ANS>. You have to return only pandas dataframe query nothing else, Query should start with 'df', not '```python\ndf' etc."
    # result_2 = llm2.invoke(instruction)
    print(result_1)
    result_df = eval(result_1.content)
    print(result_df)

    instruction_2 = f"You will receive a question enclosed in <QUESTION></QUESTION> tags along with a Pandas dataframe answer within <DATAFRAME></DATAFRAME> tags. Your task is to generate a well-structured final answer based on the user's question. <QUESTION>{query}</QUESTION>, <DATAFRAME>{result_df}</DATAFRAME>. If the data pertains to a monetary amount, include the '$' sign only if it is either negative or positive. you have to return only proper answer, not extra information or example. here is an example output: {eg_ans}, from query: {eg_qu} and dataframe: {eg_df}."
    result_3 = llm2.invoke(instruction_2)
    print("res2",result_3)
    return result_3




# Human: which product has highest quantity sold? mention product name and quantity also.

# Assistant: df.loc[(df['QUANTITY'] > 0) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_QUANTITY=('QUANTITY', 'sum')).nlargest(1, 'TOTAL_QUANTITY').round(2)

# Human: what is total paid out in second month?

# Assistant: df.loc[(df['PERIOD'] == 202202) & (df['PR_GL_ID'] == 'G103201'), 'NET_LINE_AMOUNT'].sum().round(2)

# Human: what is total paid out in third week?

# Assistant: df.loc[(df['DATE'].dt.isocalendar().week == 3) & (df['PR_GL_ID'] == 'G103201'), 'NET_LINE_AMOUNT'].sum().round(2)

# Human: which month has highest paid out?

# Assistant: df.loc[df['PR_GL_ID'] == 'G103201'].groupby('PERIOD').agg(TOTAL_PAID_OUT=('NET_LINE_AMOUNT', 'sum')).idxmax().round(2)

# Human: first month total sales return?

# Assistant: df.loc[(df['QUANTITY'] < 0) & (df['PERIOD'] == 202201), 'TOTAL_LINE_AMOUNT'].sum().round(2)

# Human: what is first month inter store sales?

# Assistant: df.loc[(df['PERIOD'] == 202201) & (df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (df['SALES_PERSON_ID'].isin([999, 145, 1])), 'NET_LINE_AMOUNT'].sum().round(2)

# Huamn: what is first week inter store sales amount?

# Assistant: df.loc[(df['DATE'].dt.isocalendar().week == 1) & (df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (df['SALES_PERSON_ID'].isin([999, 145, 1])), 'NET_LINE_AMOUNT'].sum().round(2)

# Human: which customer name did highest inter store sales in first month?

# Assistant: df.loc[(df['PERIOD'] == 202201) & (df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1])) & (df['SALES_PERSON_ID'].isin([999, 145, 1])), 'NET_LINE_AMOUNT'].sum().round(2), df.loc[(df['PERIOD'] == 202201) & (df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))].groupby(['CUSTOMER_ID', 'CUSTOMER_NAME']).agg(TOTAL_INTER_STORE_SALES=('NET_LINE_AMOUNT', 'sum')).idxmax().round(2)

# Human: which product has highest sold based on sales value?

# Assistant: df.loc[(df['QUANTITY'] > 0) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_SALES_VALUE=('TOTAL_LINE_AMOUNT', 'sum')).nlargest(1, 'TOTAL_SALES_VALUE').round(2)

# Human: which product has sold highest in first week?

# Assistant: df.loc[(df['DATE'].dt.isocalendar().week == 1) & (df['QUANTITY'] > 0) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_SALES_VALUE=('TOTAL_LINE_AMOUNT', 'sum')).nlargest(1, 'TOTAL_SALES_VALUE').round(2)

# Human: which product has highestsold based on sales value?

# Assistant: df.loc[(df['QUANTITY'] > 0) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500']))].groupby(['PR_GL_ID', 'PR_GL_DESCRIPTION']).agg(TOTAL_SALES_VALUE=('TOTAL_LINE_AMOUNT', 'sum')).nlargest(1, 'TOTAL_SALES_VALUE').round(2)

# Human: what is first month total sales?

# Assistant: df.loc[(df['PERIOD'] == 202201) & (df['QUANTITY'] > 0) & (~df['PR_GL_ID'].isin(['G103201', 'G103301', 'G021000', 'G046121', 'G021301', 'G055501', 'G064800', 'G061450', 'G042500'])) & (~df['CUSTOMER_ID'].isin([40001, 160001, 40013, 160007, 40007, 7, 1]))]['TOTAL_LINE_AMOUNT'].sum().round(2)
