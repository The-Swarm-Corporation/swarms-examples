# Import the OpenAIChat model and the Agent struct
from swarms import Agent
from swarms_memory import ChromaDB

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")


# Initilaize the chromadb client
chromadb = ChromaDB(
    metric="cosine",
    output_dir="scp",
    docs_folder="artifacts",
)

## Initialize the workflow
agent = Agent(
    agent_name = "Long-Term-Memory-Agent",
    model_name="gpt-4o-mini",
    name="Health and Wellness Blog",
    system_prompt="Generate a 10,000 word blog on health and wellness.",
    max_loops=1,
    autosave=True,
    dashboard=True,
    long_term_memory=[chromadb],
)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")
