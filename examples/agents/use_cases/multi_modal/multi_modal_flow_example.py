from swarms import Agent

from swarm_models import GPT4VisionAPI

llm = GPT4VisionAPI()

agent = Agent(
    max_loops="auto",
    model_name="gpt-4o-mini",
)

agent.run(
    task="Describe this image in a few sentences: ",
    img="https://unsplash.com/photos/0pIC5ByPpZY",
)
