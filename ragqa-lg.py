from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
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
llm = OpenAI(model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 5. Ask a question
query = "Summarize the story about assembly election in Haryana."
answer = qa_chain({"query": query})
print(answer)
