from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

# Local docs to simulate retrieval tool behavior
local_docs = {
    "What is AI?": "AI stands for Artificial Intelligence, machines imitating human intelligence.",
    "Who developed LangChain?": "LangChain was developed by Richard Vlasov and contributors to simplify LLM workflows.",
}

# Define a simple retrieval tool to use inside the retrieval agent
def retrieval_tool_func(question: str) -> str:
    return local_docs.get(question, "No information found.")

retrieval_tool = Tool(
    name="LocalDocSearch",
    func=retrieval_tool_func,
    description="Searches local documents for exact question matches"
)

# Prompt template for retrieval agent: just ask the question and return the tool result
retrieval_prompt = PromptTemplate(
    input_variables=["input"],
    template="{input}"
)

# Create the retrieval agent with single action using OpenAI LLM
retrieval_llm = OpenAI(temperature=0)
retrieval_agent_chain = LLMChain(llm=retrieval_llm, prompt=retrieval_prompt)

class SimpleRetrievalAgent(LLMSingleActionAgent):
    def plan(self, intermediate_steps):
        # Directly call retrieval tool with input as question
        return AgentAction(tool="LocalDocSearch", tool_input=intermediate_steps["input"], log="")

# Compose the retrieval agent executor with tool
retrieval_agent = AgentExecutor.from_agent_and_tools(
    agent=SimpleRetrievalAgent(
        llm_chain=retrieval_agent_chain,
        allowed_tools=["LocalDocSearch"],
    ),
    tools=[retrieval_tool],
    verbose=True,
)

# Summarization prompt and chain
summary_prompt = PromptTemplate(
    input_variables=["info"],
    template="Summarize this information concisely:\n{info}"
)
summary_llm = OpenAI(temperature=0)
summary_chain = LLMChain(llm=summary_llm, prompt=summary_prompt)

# Summarizer agent: A simple chain as an AgentExecutor for demonstration
class SummarizerAgent(LLMSingleActionAgent):
    def plan(self, intermediate_steps):
        # Just wrap info in next call (simulate single step summarization)
        return AgentFinish(return_values={"output": intermediate_steps["input"]}, log="")

summarizer_agent = AgentExecutor.from_agent_and_tools(
    agent=SummarizerAgent(
        llm_chain=summary_chain,
        allowed_tools=[],
    ),
    tools=[],
    verbose=True,
)

# Orchestrator function chaining the two agents
def multi_agent_collaboration(question: str) -> str:
    retrieved = retrieval_agent.run(question)
    summary = summarizer_agent.run(retrieved)
    return summary

if __name__ == "__main__":
    question = "What is AI?"
    answer = multi_agent_collaboration(question)
    print("Question:", question)
    print("Answer:", answer)
