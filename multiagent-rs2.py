from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Simulate retrieval with a local document list
local_docs = {
    "What is AI?": "AI stands for Artificial Intelligence, which is the capability of machines to imitate human intelligence.",
    "Who developed LangChain?": "LangChain was developed by Richard Vlasov and contributors to simplify working with LLMs.",
}

# Retrieval agent function (simulated)
def retrieval_agent(question: str) -> str:
    # Basic exact-match retrieval from local docs
    return local_docs.get(question, "Sorry, I have no info on that.")

# Summarization agent using LangChain OpenAI LLM
def summarization_agent(text: str) -> str:
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["info"],
        template="Summarize this information concisely:\n{info}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(info=text)
    return result

# Orchestrator
def multi_agent_collaboration(question: str) -> str:
    retrieved_info = retrieval_agent(question)
    summary = summarization_agent(retrieved_info)
    return summary

if __name__ == "__main__":
    question = "What is AI?"
    answer = multi_agent_collaboration(question)
    print("Question:", question)
    print("Answer:", answer)
