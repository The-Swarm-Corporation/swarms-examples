from swarms import Agent

## Initialize the workflow
agent = Agent(
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
)

# Run the workflow on a task
agent("How can we analyze blood samples to detect diabetes?")
