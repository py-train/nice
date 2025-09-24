from langgraph import Graph, State, Step
from langchain.llms import OpenAI
from langchain.agents import Tool

# Simulated local documents for retrieval
local_docs = {
    "What is AI?": "AI stands for Artificial Intelligence, machines imitating human intelligence.",
    "Who developed LangChain?": "LangChain was developed by Richard Vlasov and contributors to simplify LLM workflows.",
}

# Define retrieval tool function
def retrieval_tool_func(input_text: str) -> str:
    return local_docs.get(input_text, "No information found.")

# Create LangChain Tool for retrieval
retrieval_tool = Tool(
    name="LocalDocSearch",
    func=retrieval_tool_func,
    description="Searches local documents for exact question matches"
)

# Initialize OpenAI LLM
llm = OpenAI(temperature=0)

# Define retrieval step using the tool
retrieval_step = Step(
    id="retrieval_step",
    run_fn=lambda inputs: retrieval_tool.func(inputs["question"]),
    inputs={"question": State()},
    outputs={"retrieved_info": State()}
)

# Define summarization step using the LLM
def summarize_fn(inputs):
    prompt = f"Summarize the following information concisely:\n{inputs['retrieved_info']}"
    return llm(prompt)

summarization_step = Step(
    id="summarization_step",
    run_fn=summarize_fn,
    inputs={"retrieved_info": retrieval_step.outputs["retrieved_info"]},
    outputs={"summary": State()}
)

# Define graph connecting steps
graph = Graph(
    steps=[retrieval_step, summarization_step],
    inputs={"question": State()},
    outputs={"summary": summarization_step.outputs["summary"]}
)

def multi_agent_langgraph(question: str) -> str:
    result = graph.run({"question": question})
    return result["summary"]

if __name__ == "__main__":
    question = "What is AI?"
    answer = multi_agent_langgraph(question)
    print("Question:", question)
    print("Answer:", answer)
