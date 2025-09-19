# Install necessary libraries
# pip install chromadb sentence-transformers

from sentence_transformers import SentenceTransformer
import chromadb

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text documents to index
documents = [
    "The cat sits on the mat.",
    "A dog is playing in the garden.",
    "The sun is bright today.",
    "I love reading books on science and technology.",
    "Artificial intelligence is transforming many industries."
]

# Generate embeddings for documents
embeddings = model.encode(documents)

# Initialize ChromaDB client
client = chromadb.Client()

# Create or get collection
collection = client.create_collection(name="demo_collection")

# Add documents and their embeddings to collection
for i, (doc, emb) in enumerate(zip(documents, embeddings)):
    collection.add(
        documents=[doc],
        embeddings=[emb.tolist()],
        ids=[str(i)]
    )

# Example query
query = "Tell me about AI and technology."
query = "Pets give me peace"
query = "I love playing with pets!"
query = "The kittens love playing with the yarn!"

# Generate query embedding
query_emb = model.encode([query])[0]

# Perform similarity search to retrieve top 3 most similar documents
results = collection.query(
    query_embeddings=[query_emb.tolist()],
    n_results=3
)

print("Query:", query)
print("Top matches:")
for doc, score in zip(results['documents'][0], results['distances'][0]):
    print(f" - {doc} (Score: {score})")
