from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.schema import HumanMessage

# Simulate first agent (information retriever) by prompting
def retrieval_agent_query(question: str) -> str:
    # For simplicity, emulate retrieval behavior with predefined answers
    knowledge = {
        "What is AI?": "AI is artificial intelligence, the simulation of human intelligence by computers.",
        "Who developed LangChain?": "LangChain was developed by Richard Vlasov and his team.",
    }
    return knowledge.get(question, "Sorry, I don't have information on that.")

# Second agent (summarizer) uses OpenAI's LLM to summarize
def summarizer_agent(text: str) -> str:
    llm = OpenAI(temperature=0)
    prompt = f"Summarize the following information:\n{text}"
    response = llm([HumanMessage(content=prompt)])
    return response.content

# Orchestrator function to simulate LangGraph controlling agents
def multi_agent_collaboration(question: str) -> str:
    # Agent 1: Retrieve information
    info = retrieval_agent_query(question)
    # Agent 2: Summarize information
    summary = summarizer_agent(info)
    return summary

# Example run
if __name__ == "__main__":
    question = "What is AI?"
    result = multi_agent_collaboration(question)
    print("Question:", question)
    print("Answer (Summary):", result)
