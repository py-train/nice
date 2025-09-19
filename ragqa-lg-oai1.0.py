# Setup: Install required packages (if necessary)
# pip install langgraph langchain-openai langchain-community langchain-text-splitters chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langgraph.types import Command

# 1. Load and preprocess documents
loader = PyPDFLoader("data/iess403.pdf")
docs = loader.load()

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 3. Create embeddings and index in vector store
embedder = OpenAIEmbeddings()
vector_store = Chroma.from_documents(splits, embedder)
retriever = vector_store.as_retriever()

# 4. Wrap retriever as a tool for agent use
retriever_tool = Tool(
    name="retrieve_context",
    func=lambda query: retriever.invoke(query),
    description="Retrieves relevant chunks from indexed documents for answering queries."
)

# 5. Build LangGraph agent node
from langgraph.graph import MessagesState, StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o")

def agent_node(state):
    # Bind access to retriever_tool for LLM to decide if retrieval is needed
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

# 6. Build graph: agentic flow
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("retrieve", retriever_tool)
def agent_router(state):
    # Route: if LLM issues a tool call, go to retrieval; else, finish
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return {"retrieve": "retrieve"}
    return {END: END}
graph.add_conditional_edges("agent", agent_router, {"retrieve": "retrieve", END: END})
graph.add_edge("retrieve", "agent")
graph.add_edge(START, "agent")
graph = graph.compile()

# 7. Run the LangGraph agent for Q&A
from langchain_core.messages import HumanMessage

question = "Who won the haryana assembly elections?"
result = graph.invoke(Command({"messages": [HumanMessage(content=question)]}))
print(result["messages"][-1].content)
