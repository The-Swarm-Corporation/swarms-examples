import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from swarm_models.base_llm import BaseLLM
from swarm_models.openai_function_caller import OpenAIFunctionCaller
from swarms import Agent
from swarms.structs.agent_registry import AgentRegistry
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.concat import concat_strings
from swarms.structs.conversation import Conversation


class AgentSpec(BaseModel):
    """
    A class representing the specifications of an agent.

    Attributes:
        agent_name (str): The name of the agent.
        system_prompt (str): The system prompt for the agent.
        agent_description (str): The description of the agent.
        task (str): The main task for the agent.
        id (str): A unique identifier for the agent.
        timestamp (str): The timestamp when the agent was created.
    """

    agent_name: str = Field(..., description="The name of the agent.")
    system_prompt: str = Field(
        ...,
        description="The system prompt for the agent. Write an extremely detailed system prompt for the agent.",
    )
    agent_description: str = Field(
        ..., description="The description of the agent."
    )
    task: str = Field(..., description="The main task for the agent.")
    id: str = Field(
        default=str(uuid.uuid4()),
        description="A unique identifier for the agent.",
    )
    timestamp: str = Field(
        default=str(time.time()),
        description="The timestamp when the agent was created.",
    )


class HierarchicalOrderCall(BaseModel):
    """
    A class representing an order call for a hierarchical agent.

    Attributes:
        agent_name (str): The name of the agent to assign the task to.
        task (str): The main specific task to be assigned to the agent.
        id (str): A unique identifier for the order call.
        timestamp (str): The timestamp when the order call was created.
    """

    agent_name: str = Field(
        ...,
        description="The name of the agent to assign the task to.",
    )
    task: str = Field(
        ...,
        description="The main specific task to be assigned to the agent. Be very specific and direct.",
    )
    id: str = Field(
        default=str(uuid.uuid4()),
        description="A unique identifier for the order call.",
    )
    timestamp: str = Field(
        default=str(time.time()),
        description="The timestamp when the order call was created.",
    )


class CallTeam(BaseModel):
    """
    A class representing a call to a team of agents.

    Attributes:
        rules (str): The rules for all the agents in the swarm.
        plan (str): The plan for the swarm.
        orders (List[HierarchicalOrderCall]): The list of orders for the agents.
        id (str): A unique identifier for the call team.
        timestamp (str): The timestamp when the call team was created.
    """

    rules: str = Field(
        ...,
        description="The rules for all the agents in the swarm: e.g., All agents must return code. Be very simple and direct",
    )
    plan: str = Field(
        ...,
        description="The plan for the swarm: e.g., First create the agents, then assign tasks, then monitor progress",
    )
    orders: List[HierarchicalOrderCall]
    id: str = Field(
        default=str(uuid.uuid4()),
        description="A unique identifier for the call team.",
    )
    timestamp: str = Field(
        default=str(time.time()),
        description="The timestamp when the call team was created.",
    )


class SwarmSpec(BaseModel):
    """
    A class representing the specifications of a swarm of agents.

    Attributes:
        swarm_name (str): The name of the swarm.
        multiple_agents (List[AgentSpec]): The list of agents in the swarm.
        rules (str): The rules for all the agents in the swarm.
        plan (str): The plan for the swarm.
        id (str): A unique identifier for the swarm.
        timestamp (str): The timestamp when the swarm was created.
    """

    swarm_name: str = Field(
        ...,
        description="The name of the swarm: e.g., 'Marketing Swarm' or 'Finance Swarm'",
    )
    multiple_agents: List[AgentSpec]
    rules: str = Field(
        ...,
        description="The rules for all the agents in the swarm: e.g., All agents must return code. Be very simple and direct",
    )
    plan: str = Field(
        ...,
        description="The plan for the swarm: e.g., First create the agents, then assign tasks, then monitor progress",
    )
    id: str = Field(
        default=str(uuid.uuid4()),
        description="A unique identifier for the swarm.",
    )
    timestamp: str = Field(
        default=str(time.time()),
        description="The timestamp when the swarm was created.",
    )


HIERARCHICAL_AGENT_SYSTEM_PROMPT = """
You are a Director Boss Agent responsible for orchestrating a swarm of worker agents. Your primary duty is to serve the user efficiently, effectively, and skillfully. You dynamically create new agents when necessary or utilize existing agents, assigning them tasks that align with their capabilities. You must ensure that each agent receives clear, direct, and actionable instructions tailored to their role.

Key Responsibilities:
1. Task Delegation: Assign tasks to the most relevant agent. If no relevant agent exists, create a new one with an appropriate name and system prompt.
2. Efficiency: Ensure that tasks are completed swiftly and with minimal resource expenditure.
3. Clarity: Provide orders that are simple, direct, and actionable. Avoid ambiguity.
4. Dynamic Decision Making: Assess the situation and choose the most effective path, whether that involves using an existing agent or creating a new one.
5. Monitoring: Continuously monitor the progress of each agent and provide additional instructions or corrections as necessary.

Instructions:
- Identify the Task: Analyze the input task to determine its nature and requirements.
- Agent Selection/Creation:
  - If an agent is available and suited for the task, assign the task to that agent.
  - If no suitable agent exists, create a new agent with a relevant system prompt.
- Task Assignment: Provide the selected agent with explicit and straightforward instructions.
- Reasoning: Justify your decisions when selecting or creating agents, focusing on the efficiency and effectiveness of task completion.
"""


class HierarchicalAgentSwarm(BaseSwarm):
    """
    A class to create and manage a hierarchical swarm of agents.

    Attributes:
        name (str): The name of the swarm.
        description (str): A description of the swarm.
        director (Any): The director agent.
        agents (List[Agent]): The list of worker agents.
        max_loops (int): The maximum number of loops to run.
        create_agents_on (bool): Whether to create agents on the fly.
        template_worker_agent (Agent): A template for worker agents.
        director_planning_prompt (str): The planning prompt for the director.
        template_base_worker_llm (BaseLLM): The base language model for worker agents.
        swarm_history (str): The history of the swarm's actions.
        agent_registry (AgentRegistry): The registry of agents.
        conversation (Conversation): The conversation history.

    Methods:
        agents_check(): Check if the agents are properly set up.
        add_agents_into_registry(agents: List[Agent]): Add agents to the registry.
        create_agent(agent_name: str, system_prompt: str, agent_description: str, task: str = None) -> str: Create a single agent.
        parse_json_for_agents_then_create_agents(function_call: dict) -> str: Parse JSON and create or run agents.
        run(task: str) -> str: Run the swarm on a given task.
        run_new(task: str): Run the swarm with new agent creation.
        check_agent_output_type(response: Any): Check and convert the agent's output type.
        distribute_orders_to_agents(order_dict: dict) -> str: Distribute orders to agents and run them.
        create_single_agent(name: str, system_prompt: str, description: str) -> Agent: Create a single agent.
        create_agents_from_func_call(function_call: dict): Create agents from a function call.
        plan(task: str) -> str: Plan the tasks for the agents.
        log_director_function_call(function_call: dict): Log the director's function call.
        run_worker_agent(name: str = None, task: str = None, *args, **kwargs): Run a worker agent.
        list_agents(): List all available agents.
        list_agents_available() -> str: Get a string representation of available agents.
        find_agent_by_name(agent_name: str = None, *args, **kwargs) -> Optional[Agent]: Find an agent by name.
    """

    def __init__(
        self,
        name: str = "HierarchicalAgentSwarm",
        description: str = "A swarm of agents that can be used to distribute tasks to a team of agents.",
        director: Any = None,
        agents: List[Agent] = None,
        max_loops: int = 1,
        create_agents_on: bool = False,
        template_worker_agent: Agent = None,
        director_planning_prompt: str = None,
        template_base_worker_llm: BaseLLM = None,
        swarm_history: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the HierarchicalAgentSwarm.

        Args:
            name (str): The name of the swarm.
            description (str): A description of the swarm.
            director (Any): The director agent.
            agents (List[Agent]): The list of worker agents.
            max_loops (int): The maximum number of loops to run.
            create_agents_on (bool): Whether to create agents on the fly.
            template_worker_agent (Agent): A template for worker agents.
            director_planning_prompt (str): The planning prompt for the director.
            template_base_worker_llm (BaseLLM): The base language model for worker agents.
            swarm_history (str): The history of the swarm's actions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.director = director
        self.agents = agents or []
        self.max_loops = max_loops
        self.create_agents_on = create_agents_on
        self.template_worker_agent = template_worker_agent
        self.director_planning_prompt = director_planning_prompt
        self.template_base_worker_llm = template_base_worker_llm
        self.swarm_history = swarm_history

        self.agents_check()
        self.agent_registry = AgentRegistry()
        self.add_agents_into_registry(self.agents)
        self.conversation = Conversation(time_enabled=True)
        self.swarm_history = (
            self.conversation.return_history_as_string()
        )

    def agents_check(self) -> None:
        """
        Check if the agents are properly set up.

        Raises:
            ValueError: If the director is not set, no agents are available, or max_loops is 0.
        """
        if self.director is None:
            raise ValueError("The director is not set.")

        if len(self.agents) == 0:
            self.create_agents_on = True

        if len(self.agents) > 0:
            self.director.base_model = CallTeam
            self.director.system_prompt = (
                HIERARCHICAL_AGENT_SYSTEM_PROMPT
            )

        if self.max_loops == 0:
            raise ValueError("The max_loops is not set.")

    def add_agents_into_registry(self, agents: List[Agent]) -> None:
        """
        Add agents into the agent registry.

        Args:
            agents (List[Agent]): A list of agents to add into the registry.
        """
        for agent in agents:
            self.agent_registry.add(agent)

    def create_agent(
        self,
        agent_name: str,
        system_prompt: str,
        agent_description: str,
        task: str = None,
    ) -> str:
        """
        Creates an individual agent.

        Args:
            agent_name (str): The name of the agent.
            system_prompt (str): The system prompt for the agent.
            agent_description (str): The description of the agent.
            task (str, optional): The task for the agent. Defaults to None.

        Returns:
            str: The output of the agent's run.
        """
        logger.info(f"Creating agent: {agent_name}")

        agent = Agent(
            agent_name=agent_name,
            llm=self.template_base_worker_llm,
            system_prompt=system_prompt,
            agent_description=agent_description,
            retry_attempts=1,
            verbose=False,
            dashboard=False,
        )

        self.agents.append(agent)

        logger.info(f"Running agent: {agent_name} on task: {task}")
        output = agent.run(task)

        self.conversation.add(role=agent_name, content=output)
        return output

    def parse_json_for_agents_then_create_agents(
        self, function_call: dict
    ) -> str:
        """
        Parses a JSON function call to create or run a list of agents.

        Args:
            function_call (dict): The JSON function call specifying the agents.

        Returns:
            str: A concatenated string of agent responses.
        """
        responses = []
        logger.info("Parsing JSON for agents")

        if self.create_agents_on:
            for agent in function_call["multiple_agents"]:
                out = self.create_agent(
                    agent_name=agent["agent_name"],
                    system_prompt=agent["system_prompt"],
                    agent_description=agent["agent_description"],
                    task=agent["task"],
                )
                responses.append(out)
        else:
            for agent in function_call["orders"]:
                out = self.run_worker_agent(
                    name=agent["agent_name"],
                    task=agent["task"],
                )
                responses.append(out)

        return concat_strings(responses)

    def run(self, task: str) -> str:
        """
        Runs the function caller to create and execute agents based on the provided task.

        Args:
            task (str): The task for which the agents need to be created and executed.

        Returns:
            str: The output of the agents' execution.
        """
        logger.info("Running the swarm")

        function_call = self.director.run(task)
        self.conversation.add(
            role="Director", content=str(function_call)
        )
        self.log_director_function_call(function_call)

        return self.parse_json_for_agents_then_create_agents(
            function_call
        )

    def run_new(self, task: str) -> str:
        """
        Runs the function caller to create and execute agents based on the provided task.

        Args:
            task (str): The task for which the agents need to be created and executed.

        Returns:
            str: The output of the agents' execution.
        """
        logger.info("Running the swarm")

        function_call = self.director.run(task)
        self.conversation.add(
            role="Director", content=str(function_call)
        )
        self.log_director_function_call(function_call)

        if self.create_agents_on:
            self.create_agents_from_func_call(function_call)
            self.director.base_model = CallTeam

            orders_prompt = f"Now, the agents have been created. Submit orders to the agents to enable them to complete the task: {task}: {self.list_agents_available()}"
            orders = self.director.run(orders_prompt)
            self.conversation.add(
                role="Director",
                content=str(orders_prompt + str(orders)),
            )

            orders = self.check_agent_output_type(orders)
            return self.distribute_orders_to_agents(orders)

    def check_agent_output_type(self, response: Any) -> Dict:
        """
        Check and convert the agent's output type.

        Args:
            response (Any): The response from the agent.

        Returns:
            Dict: The response converted to a dictionary.
        """
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            return eval(response)
        else:
            return response

    def distribute_orders_to_agents(self, order_dict: dict) -> str:
        """
        Distribute orders to agents and run them.

        Args:
            order_dict (dict): The dictionary containing orders for agents.

        Returns:
            str: A concatenated string of agent responses.
        """
        responses = []

        for order in order_dict["orders"]:
            agent_name = order["agent_name"]
            task = order["task"]

            response = self.run_worker_agent(
                name=agent_name, task=task
            )

            log = f"Agent: {agent_name} completed task: {task} with response: {response}"
            self.conversation.add(
                role=agent_name, content=task + response
            )
            responses.append(log)
            logger.info(log)

        return concat_strings(responses)

    def create_single_agent(
        self, name: str, system_prompt: str, description: str
    ) -> Agent:
        """
        Create a single agent from the agent specification.

        Args:
            name (str): The name of the agent.
            system_prompt (str): The system prompt for the agent.
            description (str): The description of the agent.

        Returns:
            Agent: The created agent.
        """
        agent = Agent(
            agent_name=name,
            llm=self.template_base_worker_llm,
            system_prompt=system_prompt,
            agent_description=description,
            max_loops=1,
            retry_attempts=1,
            verbose=False,
            dashboard=False,
        )

        self.agents.append(agent)
        return agent

    def create_agents_from_func_call(
        self, function_call: dict
    ) -> None:
        """
        Create agents from the function call.

        Args:
            function_call (dict): The function call containing the agent specifications.
        """
        logger.info("Creating agents from the function call")
        for agent_spec in function_call["multiple_agents"]:
            agent = self.create_single_agent(
                name=agent_spec["agent_name"],
                system_prompt=agent_spec["system_prompt"],
                description=agent_spec["agent_description"],
            )
            logger.info(
                f"Created agent: {agent.agent_name} with description: {agent.description}"
            )

    def plan(self, task: str) -> str:
        """
        Plans the tasks for the agents in the swarm.

        Args:
            task (str): The task to be planned.

        Returns:
            str: The planned task for the agents.
        """
        logger.info("Director is planning the task")
        self.director.system_prompt = self.director_planning_prompt
        return self.director.run(task)

    def log_director_function_call(self, function_call: dict) -> None:
        """
        Log the director's function call.

        Args:
            function_call (dict): The function call to be logged.
        """
        logger.info(f"Swarm Name: {function_call['swarm_name']}")
        logger.info(f"Plan: {function_call['plan']}")
        logger.info(
            f"Number of agents: {len(function_call['multiple_agents'])}"
        )

        for agent in function_call["multiple_agents"]:
            logger.info(f"Agent: {agent['agent_name']}")
            logger.info(f"Description: {agent['agent_description']}")

    def run_worker_agent(
        self, name: str = None, task: str = None, *args, **kwargs
    ) -> str:
        """
        Run the worker agent.

        Args:
            name (str, optional): The name of the worker agent. Defaults to None.
            task (str, optional): The task to send to the worker agent. Defaults to None.

        Returns:
            str: The response from the worker agent.

        Raises:
            Exception: If an error occurs while running the worker agent.
        """
        try:
            agent = self.find_agent_by_name(name)
            response = agent.run(task, *args, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def list_agents(self) -> None:
        """
        List all available agents in the swarm.
        """
        logger.info("Listing agents available in the swarm")
        for agent in self.agents:
            name = agent.agent_name
            description = (
                agent.description or "No description available."
            )
            logger.info(f"Agent: {name}, Description: {description}")

    def list_agents_available(self) -> str:
        """
        Get a string representation of available agents.

        Returns:
            str: A formatted string listing all available agents.
        """
        number_of_agents_available = len(self.agents)
        agent_list = "\n".join(
            [
                f"Agent {agent.agent_name}: Description {agent.description}"
                for agent in self.agents
            ]
        )
        return f"""
        There are currently {number_of_agents_available} agents available in the swarm.

        Agents Available:
        {agent_list}
        """

    def find_agent_by_name(
        self, agent_name: str = None, *args, **kwargs
    ) -> Optional[Agent]:
        """
        Finds an agent in the swarm by name.

        Args:
            agent_name (str, optional): The name of the agent to find. Defaults to None.

        Returns:
            Optional[Agent]: The agent with the specified name, or None if not found.
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None


def create_agents_with_boss(task: str) -> dict:
    """
    Create agents using the boss agent based on the given task.

    Args:
        task (str): The task description for creating agents.

    Returns:
        dict: The output containing the created agents and their specifications.
    """
    model = OpenAIFunctionCaller(
        system_prompt=HIERARCHICAL_AGENT_SYSTEM_PROMPT,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_model=SwarmSpec,
        max_tokens=5000,
    )

    out = model.run(task)
    logger.info(f"Boss agent output: {out}")

    # Save the output to a file
    with open(
        f"{uuid.uuid4().hex}_agent_creation_dict.json", "w"
    ) as f:
        json.dump(out, f, indent=4)

    return out


def design_and_run_swarm(task: str) -> str:
    """
    Design and run a swarm of agents based on the given task.

    Args:
        task (str): The task description for the swarm.

    Returns:
        str: The output of the swarm execution.
    """
    logger.info("Creating agents with the boss agent.")
    agents_dict = create_agents_with_boss(task)
    task_for_agent = agents_dict.get("task")

    logger.info("Creating HierarchicalAgentSwarm.")
    swarm = HierarchicalAgentSwarm(
        name=agents_dict["swarm_name"],
        description=f"Swarm for task: {task}",
        director=Agent(
            agent_name="Director",
            system_prompt=HIERARCHICAL_AGENT_SYSTEM_PROMPT,
            llm=OpenAIFunctionCaller(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_model=SwarmSpec,
                max_tokens=5000,
            ),
        ),
        create_agents_on=True,
        template_base_worker_llm=OpenAIFunctionCaller(
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000,
        ),
    )

    logger.info("Running the swarm.")
    return swarm.run_new(task_for_agent)


if __name__ == "__main__":
    # Configure logging
    logger.add("swarm_execution.log", rotation="500 MB")

    # Example usage
    task = """
    Create a swarm of agents specialized in every social media platform to market the swarms github framework 
    which makes it easy for you to orchestrate and manage multiple agents in a swarm.
    Create a minimum of 10 agents that are hyper-specialized in different areas of social media marketing.

    We need to promote the new SpreadSheet Swarm feature that allows you to run multiple agents in parallel 
    and manage them from a single dashboard.
    Here is the link: https://docs.swarms.world/en/latest/swarms/structs/spreadsheet_swarm/
    """

    result = design_and_run_swarm(task)
    print(result)
