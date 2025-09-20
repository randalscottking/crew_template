import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai_tools import (
    CSVSearchTool,
    DirectoryReadTool,
    FileReadTool,
    RagTool,
    SerperDevTool,
)
from dotenv import load_dotenv


class CrewManager:
    """A manager class for organizing and controlling CrewAI workflows."""

    def __init__(self):
        """Initialize the CrewManager with environment setup and empty containers."""
        self.load_environment()
        self.tools = self.initialize_tools()
        self.agents = []
        self.tasks = []
        self.crew = None

    def load_environment(self):
        """Load environment variables from .env file and set OpenAI configuration."""
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

    def initialize_tools(self) -> Dict[str, Any]:
        """Initialize and return a dictionary of available CrewAI tools."""
        return {
            "file_tool": FileReadTool(),
            "search_tool": SerperDevTool(),
            "csv_search_tool": CSVSearchTool(),
            "rag_tool": RagTool(),
        }

    def create_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[Any] = None,
        verbose: bool = False,
    ) -> Agent:
        """Create a new agent with specified parameters and add it to the agents list."""
        if tools is None:
            tools = []
        agent = Agent(
            role=role, goal=goal, backstory=backstory, verbose=verbose, tools=tools
        )
        self.agents.append(agent)
        return agent

    def create_task(self, name: str, description: str, tools: List[Any] = None) -> Task:
        """Create a new task with specified parameters and add it to the tasks list."""
        if tools is None:
            tools = []
        task = Task(name=name, description=description, tools=tools)
        self.tasks.append(task)
        return task

    def create_crew(
        self,
        process: Process = Process.hierarchical,
        manager_llm: str = "gpt-4o",
        verbose: bool = True,
        planning: bool = True,
    ) -> Crew:
        """Create a crew from existing agents and tasks with specified configuration."""
        self.crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=verbose,
            planning=planning,
            process=process,
            manager_llm=manager_llm,
        )
        return self.crew

    def run_crew(self):
        """Execute the crew workflow and return the results."""
        if self.crew is None:
            raise ValueError("Crew not created. Call create_crew() first.")
        return self.crew.kickoff()

    def get_user_input(self) -> str:
        """Display welcome message and prompt user for input."""
        print("Welcome!")
        return input("What is your question? ")


def main():
    """Main function to demonstrate CrewManager usage with example agents and tasks."""
    crew_manager = CrewManager()

    # Create agents
    crew_manager.create_agent(
        role="Research Analyst",
        goal="Gather comprehensive information on given topics",
        backstory="You are an experienced research analyst with expertise in data collection and analysis",
        tools=[crew_manager.tools["search_tool"], crew_manager.tools["file_tool"]],
        verbose=False
    )

    crew_manager.create_agent(
        role="Content Writer",
        goal="Create engaging and informative content based on research findings",
        backstory="You are a skilled content writer who can transform complex information into clear, readable content",
        tools=[crew_manager.tools["file_tool"]],
        verbose=False
    )

    crew_manager.create_agent(
        role="Quality Reviewer",
        goal="Review and ensure the quality and accuracy of all deliverables",
        backstory="You are a meticulous quality reviewer with an eye for detail and accuracy",
        tools=[crew_manager.tools["file_tool"]],
        verbose=False
    )

    # Create tasks
    crew_manager.create_task(
        name="Research Task",
        description="Conduct thorough research on the user's question and gather relevant information",
        tools=[crew_manager.tools["search_tool"], crew_manager.tools["rag_tool"]]
    )

    crew_manager.create_task(
        name="Content Creation Task",
        description="Create comprehensive content based on the research findings and user requirements",
        tools=[crew_manager.tools["file_tool"]]
    )

    # Create and run crew
    crew_manager.create_crew()
    question = crew_manager.get_user_input()
    crew_manager.run_crew()


if __name__ == "__main__":
    main()
