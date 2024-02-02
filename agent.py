import os
from pathlib import Path
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.agents import initialize_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI

from modules.database.database import PostgresDB

"""
Here we define some environment variables and the tools that the agent will use.
Along with some configuration for the app to start.
"""
load_dotenv()

chunk_size = 512
chunk_overlap = 50

embeddings_model = OpenAIEmbeddings()
openai_client = AsyncOpenAI()

CSV_STORAGE_PATH = "./data"


def process_pdfs(pdf_storage_path: str):
    csv_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=50)

    for csv_path in csv_directory.glob("*.csv"):
        loader = CSVLoader(file_path=str(csv_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    documents_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        documents_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return documents_search


doc_search = process_pdfs(CSV_STORAGE_PATH)

"""
Execute SQL query tool definition along schemas.
"""


def execute_sql(query: str) -> str:
    """
    Execute queries against the database. It needs to be a clean SQL
    query in one line without backticks or line jumps.
    """
    db = PostgresDB()
    db.connect_with_url(os.getenv("DB_URL"))
    results = db.run_sql_to_markdown(query)

    return results


class ExecuteSqlToolInput(BaseModel):
    query: str = Field(
        description="A clean SQL query in one line to be executed agains the database")


execute_sql_tool = StructuredTool(
    func=execute_sql,
    name="Execute SQL",
    description="useful for when you need to execute SQL queries against the database. Always use a clause LIMIT 10",
    args_schema=ExecuteSqlToolInput
)

"""
Research database tool definition along schemas.
"""


def research_database(user_request: str) -> str:
    """
    Searches for table definitions matching the user request
    """
    search_kwargs = {"k": 30}

    retriever = doc_search.as_retriever(search_kwargs=search_kwargs)

    def format_docs(docs):
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.page_content}")
        return "\n\n".join([d.page_content for d in docs])

    results = retriever.invoke(user_request)

    return format_docs(results)


class ResearchDatabaseToolInput(BaseModel):
    user_request: str = Field(
        description="The user query to search against the table definitions for matches. Always use a clase of LIMIT 10")


research_database_tool = StructuredTool(
    func=research_database,
    name="Search db info",
    description="Search for database information so you can have context for building SQL queries.",
    args_schema=ResearchDatabaseToolInput
)


@cl.on_chat_start
def start():
    tools = [execute_sql_tool, research_database_tool]

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
            You are a world class data scientist, your job is to listen to the user query
            and based on it, use on of your tools to do the job. Usually you would start by analyzing
            for possible SQL queries the user wants to build based on your knowledge base.
            Remember your tools are:

            - execute_sql (bring back the results as markdown table)
            - research_database (search for table definitions so you can build a SQL Query)

            Think carefully before routing to one of the tools. If you don't know what the user wants or you
            dont understand, ask for clarification.
            Remember, if you don't know the answer don't make anything up. Always ask for feedback.
            One last detail: always run the querys with LIMIT 10.

            User query: {input}
        """
    )

    agent = initialize_agent(tools=tools, prompt=prompt, llm=llm)

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    await cl.Message(content=res).send()
