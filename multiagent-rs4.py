from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType

# Simulate local documents for retrieval
local_docs = {
    "What is AI?": "AI stands for Artificial Intelligence, machines imitating human intelligence.",
    "Who developed LangChain?": "LangChain was developed by Richard Vlasov and contributors to simplify LLM workflows.",
}

# Retrieval tool function using local_docs
def local_retrieval_tool(question: str) -> str:
    return local_docs.get(question, "No information found.")

# Define retrieval tool
retrieval_tool = Tool(
    name="LocalDocSearch",
    func=local_retrieval_tool,
    description="Searches local documents for exact question matches"
)

# Initialize OpenAI LLM
llm = OpenAI(temperature=0)

# Initialize retrieval agent
retrieval_agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Summarization prompt and chain
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following information concisely:\n{text}"
)

summarization_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Summarization function
def summarization_agent(text: str) -> str:
    return summarization_chain.run(text=text)

# Orchestrator function chaining retrieval then summarization
def multi_agent_collaboration(question: str) -> str:
    retrieved_info = retrieval_agent.run(question)
    summary = summarization_agent(retrieved_info)
    return summary

if __name__ == "__main__":
    question = "What is AI?"
    answer = multi_agent_collaboration(question)
    print("Question:", question)
    print("Answer:", answer)
