from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Prompt for rephrasing the user's query
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2. Prompt for generating the final answer
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer concise.\
<context>
{context}
</context>"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory


# Existing components
llm = ChatOpenAI()
docs = [
    Document(page_content="The sun is a star at the center of our solar system."),
    Document(page_content="The moon is Earth's only natural satellite."),
    Document(page_content="Mars is known as the 'Red Planet'."),
]
vector_store = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()

# Create the history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# from langchain.memory import ChatMessageHistory
# from langchain.schema.runnable import RunnableWithMessageHistory
# deprecated /\

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create an in-memory session store
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

stuff_documents_chain = create_stuff_documents_chain(llm, qa_prompt)

retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)



# Wrap the chain with RunnableWithMessageHistory
conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key='answer'
)

# 5. Invoke the conversational chain multiple times
response1 = conversational_rag_chain.invoke(
    {"input": "What is the moon?"},
    config={"configurable": {"session_id": "unique_session_1"}}
)
print(f"User: What is the moon?")
print(f"AI: {response1}")

response2 = conversational_rag_chain.invoke(
    {"input": "What planet is it associated with?"},
    config={"configurable": {"session_id": "unique_session_1"}}
)
print(f"User: What planet is it associated with?")
print(f"AI: {response2}")
