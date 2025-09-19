from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# 1. Load and chunk documents
loader = PyPDFLoader('data/iess403.pdf')
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Create embeddings and vector store
embedder = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embedder)

# 3. Expose retriever interface
retriever = vector_store.as_retriever()

# 4. Build RetrievalQA chain
llm = ChatOpenAI(model="gpt-4o")  # or "gpt-3.5-turbo", etc.
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 5. Ask a question
while True:
    query = "Summarize the assembly election story from Haryana."
    query = input('Ask a question: ')
    answer = qa_chain.invoke({"query": query})
    print("Question:\n", answer['query'])
    print("Answer:\n", answer['result'])
