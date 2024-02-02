import os
import re
from datetime import date
from pathlib import Path
from typing import List

import chainlit as cl
from chainlit.playground.providers.openai import ChatOpenAI as ChatOpenAIProvider
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI

from modules.database.database import PostgresDB

load_dotenv()

chunk_size = 512
chunk_overlap = 50

embeddings_model = OpenAIEmbeddings()
openai_client = AsyncOpenAI()
CSV_STORAGE_PATH = "./data"

settings = {"model": "gpt-4", "temperature": 0}

explain_query_result_prompt = """
You received a SQL query result.
For context, the query was: {user_query}

Results:

{table}

Write the SQL query that was executed and below attach the results:
"""


def extract_sql_from_markdown(markdown_text):
    print("BEFORE EXTRACTION")
    print(markdown_text)
    # Define a regular expression pattern to match SQL code enclosed in backticks
    sql_pattern = r'```sql([\s\S]*?)```'

    # Use re.findall to find all matches of SQL code
    sql_code_matches = re.findall(sql_pattern, markdown_text)

    # Join the matched SQL code into a single string
    extracted_sql = '\n'.join(sql_code_matches)
    print("EXTRACTED SQL")
    print(extracted_sql)
    return extracted_sql


@cl.step
async def execute_query(sql_query: str):
    raw_query = extract_sql_from_markdown(sql_query)

    @tool
    def execute_sql(query: str) -> str:
        """
        Execute queries against the database. It needs to be a clean SQL
        query in one line without backticks or line jumps.
        """
        db = PostgresDB()
        db.connect_with_url(os.getenv("DB_URL"))
        results = db.run_sql_to_markdown(query)

        return results

    return execute_sql(raw_query)


@cl.step
async def analyze(table, user_query):
    settings = {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    current_step = cl.context.current_step
    today = str(date.today())
    current_step.generation = cl.ChatGeneration(
        provider=ChatOpenAIProvider.id,
        messages=[
            cl.GenerationMessage(
                role="user",
                template=explain_query_result_prompt,
                formatted=explain_query_result_prompt.format(
                    date=today, table=table, user_query=user_query),
            )
        ],
        settings=settings,
        inputs={"date": today, "table": table},
    )

    final_answer = cl.Message(content="")
    await final_answer.send()

    # Call OpenAI and stream the message
    stream = await openai_client.chat.completions.create(
        messages=[m.to_openai() for m in current_step.generation.messages],
        stream=True,
        **settings
    )
    async for part in stream:
        token = part.choices[0].delta.content or ""
        if token:
            await final_answer.stream_token(token)

    await final_answer.update()

    current_step.output = final_answer.content
    current_step.generation.completion = final_answer.content

    return current_step.output


@cl.step(type="llm", name="Run SQL")
async def run_sql(sql_query: str):
    table = await execute_query(sql_query)
    analysis = await analyze(table, sql_query)
    return analysis


@cl.action_callback("Run")
async def on_action(action: cl.Action):
    last_message = cl.user_session.get("root_message")

    await run_sql(last_message.content)

    return "SQL query executed and analyzed!"


def process_pdfs(pdf_storage_path: str):
    csv_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=50)

    for csv_path in csv_directory.glob("*.csv"):
        loader = CSVLoader(file_path=str(csv_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    doc_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search


doc_search = process_pdfs(CSV_STORAGE_PATH)
model = ChatOpenAI(model_name="gpt-4", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    template = """
    You are a world-class data scientist and have been hired by a company to analyze their database.
    Your job is to think carefully and write eficient and memory effective SQL queries.
    Answer the question based only on the following context.
    The context comes from a CSV listing tables and its descriptions from a database.
    Your job is to build SQL queries based on the context provided answering the question.
    Allways add a LIMIT 10 clause to not overload the server please.

    {context}

    Question: {question}

    SQL Query written in markdown based on the context provided and the Question:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # print docs in a loop with a number:

    def format_docs(docs):
        for i, doc in enumerate(docs):
            print(f"{i + 1}. {doc.page_content}")
        return "\n\n".join([d.page_content for d in docs])

    search_kwargs = {"k": 30}

    retriever = doc_search.as_retriever(search_kwargs=search_kwargs)

    runnable = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    # first we define button to execute the action:

    runnable = cl.user_session.get("runnable")  # type: Runnable

    actions = [
        cl.Action(name="Run", value=message.content,
                  description="Run SQL!")
    ]

    msg = cl.Message(content="", actions=actions)

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                # Add unique pairs to the set
                self.sources.add(source_page_pair)

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join(
                    [f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text,
                            display="inline")
                )

    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
                message.content,
                config=RunnableConfig(callbacks=[
                    cl.LangchainCallbackHandler(),
                    PostMessageHandler(msg)
                ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()
