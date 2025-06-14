from swarms import Agent

# Initialize the task
task = (
    "Analyze this image of an assembly line and identify any issues such as"
    " misaligned parts, defects, or deviations from the standard assembly"
    " process. IF there is anything unsafe in the image, explain why it is"
    " unsafe and how it could be improved."
)
img = "assembly_line.jpg"

## Initialize the workflow
agent = Agent(
    agent_name="Multi-ModalAgent",
    model_name="gpt-4o-mini",
    max_loops="auto",
    autosave=True,
    dashboard=True,
    multi_modal=True,
)

# Run the workflow on a task
agent.run(task, img)
