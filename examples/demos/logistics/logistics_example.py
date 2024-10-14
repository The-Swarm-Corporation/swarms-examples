import os

from dotenv import load_dotenv

from swarm_models import GPT4VisionAPI
from swarms.prompts.logistics import (
    Efficiency_Agent_Prompt,
    Health_Security_Agent_Prompt,
    Productivity_Agent_Prompt,
    Quality_Control_Agent_Prompt,
    Safety_Agent_Prompt,
    Security_Agent_Prompt,
    Sustainability_Agent_Prompt,
)
from swarms.structs import Agent

# from swarms.utils.banana_wrapper import banana

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# GPT4VisionAPI or llama
# @banana #- deploy to banana
llm = GPT4VisionAPI(openai_api_key=api_key)

# Image for analysis
factory_image = "factory_image1.jpg"

# Initialize agents with respective prompts
health_security_agent = Agent(
    llm=llm,
    sop=Health_Security_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

quality_control_agent = Agent(
    llm=llm,
    sop=Quality_Control_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

productivity_agent = Agent(
    llm=llm,
    sop=Productivity_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

safety_agent = Agent(
    llm=llm, sop=Safety_Agent_Prompt, max_loops=1, multi_modal=True
)

security_agent = Agent(
    llm=llm, sop=Security_Agent_Prompt, max_loops=1, multi_modal=True
)

sustainability_agent = Agent(
    llm=llm,
    sop=Sustainability_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

efficiency_agent = Agent(
    llm=llm,
    sop=Efficiency_Agent_Prompt,
    max_loops=1,
    multi_modal=True,
)

# Run agents with respective tasks on the same image
health_analysis = health_security_agent.run(
    "Analyze the safety of this factory", factory_image
)
quality_analysis = quality_control_agent.run(
    "Examine product quality in the factory", factory_image
)
productivity_analysis = productivity_agent.run(
    "Evaluate factory productivity", factory_image
)
safety_analysis = safety_agent.run(
    "Inspect the factory's adherence to safety standards",
    factory_image,
)
security_analysis = security_agent.run(
    "Assess the factory's security measures and systems",
    factory_image,
)
sustainability_analysis = sustainability_agent.run(
    "Examine the factory's sustainability practices", factory_image
)
efficiency_analysis = efficiency_agent.run(
    "Analyze the efficiency of the factory's manufacturing process",
    factory_image,
)
