from swarms.prompts.accountant_swarm_prompts import (
    DECISION_MAKING_PROMPT,
    DOC_ANALYZER_AGENT_PROMPT,
    SUMMARY_GENERATOR_AGENT_PROMPT,
)
from swarms.structs import Agent
from swarms.utils.pdf_to_text import pdf_to_text

# Agents
doc_analyzer_agent = Agent(
    model_name="gpt-4o-mini",
    sop=DOC_ANALYZER_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path="doc_analyzer_agent.json",
)
summary_generator_agent = Agent(
    model_name="gpt-4o-mini",
    sop=SUMMARY_GENERATOR_AGENT_PROMPT,
    max_loops=1,
    autosave=True,
    saved_state_path="summary_generator_agent.json",
)
decision_making_support_agent = Agent(
    model_name="gpt-4o-mini",
    sop=DECISION_MAKING_PROMPT,
    max_loops=1,
    saved_state_path="decision_making_support_agent.json",
)


pdf_path = "bankstatement.pdf"
fraud_detection_instructions = "Detect fraud in the document"
summary_agent_instructions = (
    "Generate an actionable summary of the document with action steps"
    " to take"
)
decision_making_support_agent_instructions = (
    "Provide decision making support to the business owner:"
)


# Transform the pdf to text
pdf_text = pdf_to_text(pdf_path)
print(pdf_text)


# Detect fraud in the document
fraud_detection_agent_output = doc_analyzer_agent.run(
    f"{fraud_detection_instructions}: {pdf_text}"
)

# Generate an actionable summary of the document
summary_agent_output = summary_generator_agent.run(
    f"{summary_agent_instructions}: {fraud_detection_agent_output}"
)

# Provide decision making support to the accountant
decision_making_support_agent_output = (
    decision_making_support_agent.run(
        f"{decision_making_support_agent_instructions}:"
        f" {summary_agent_output}"
    )
)
