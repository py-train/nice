# Required libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# 1. Instantiate the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. Create a retriever from a simple, in-memory vector store
# In a real application, you would load documents and use a persistent vector store.
docs = [
    Document(page_content="The sun is a star at the center of our solar system."),
    Document(page_content="The moon is Earth's only natural satellite."),
    Document(page_content="Mars is known as the 'Red Planet'."),
]
vector_store = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()

# 3. Define the prompt template for the 'stuff' chain
template = """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Create the 'stuff' document chain
# This chain takes the retrieved documents, formats them into the prompt, and passes it to the LLM.
stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

# 5. Create the full retrieval chain
# This chain combines the retriever with the document chain.
retrieval_chain = create_retrieval_chain(retriever, stuff_documents_chain)

# 6. Add the output parser to ensure the final output is a string.
full_lcel_chain = retrieval_chain | StrOutputParser()
# 7. Invoke the chain with a user query
response = full_lcel_chain.invoke({"input": "What is the moon?"})

# Print the response
print(response)

print(response['answer'])
