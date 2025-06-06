import os

from dotenv import load_dotenv

from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.company import Company

load_dotenv()

llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"), max_tokens=4000
)

ceo = Agent(model_name="gpt-4o-mini", ai_name="CEO")
dev = Agent(model_name="gpt-4o-mini", ai_name="Developer")
va = Agent(model_name="gpt-4o-mini", ai_name="VA")

# Create a company
company = Company(
    org_chart=[[dev, va]],
    shared_instructions="Do your best",
    ceo=ceo,
)

# Add agents to the company
hr = Agent(model_name="gpt-4o-mini", name="HR")
company.add(hr)

# Get an agent from the company
hr = company.get("CEO")

# Remove an agent from the company
company.remove(hr)

# Run the company
company.run()
