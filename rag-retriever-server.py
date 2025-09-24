# !! Not working code !!

CONFLUENCE_SITE = 'https://utexas.atlassian.net/wiki/'

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Confluence-retriever")



from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load documents from Confluence public space anonymously
# Set your public Confluence space URL and space key (namespace of the wiki space)
confluence_url = CONFLUENCE_SITE
confluence_space_key = "caeegraduateoffice"

# Anonymous load - omit username and api_key for publicly accessible Confluence pages
loader = ConfluenceLoader(
    url=confluence_url,
    space_key=confluence_space_key,
    limit=100,  # number of pages to fetch, adjust to your needs
    # Do NOT pass username or api_key for anonymous access
    cloud=True,
    max_pages=100,
)
documents = loader.load()

len(documents)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the following questions as best you can."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


history = InMemoryChatMessageHistory()

def get_history():
    return history


# 2. Split documents into smaller chunks for embeddings and retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 3. Initialize embeddings and vector store (Chroma)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Use OpenAI embeddings
persist_directory = "./chroma_persist"  # Local directory for persistent vector storage

# Create/load Chroma vector store with embedding function
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="confluence_public_docs"
)
# Persist to disk
vectorstore.persist()

# 4. Initialize the ChatOpenAI LLM
#     llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOpenAI(temperature=0)



# 5. Create RetrievalQA chain using vectorstore's retriever and the LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tool_wrapper = mcp.tool(description='Retrieves from public (anonymous) confluence wikis',
                        name='confluence retriever'
                       )

    
qa_chain = tool_wrapper(qa_chain)



if __name__ == '__main__':
    mcp.run('http')



# qa_chain_tool = MCPTool(
#     name = 'confl retr',
#     description = 'Retrieves from public (anonymous) confluence wikis'
