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
    def __init__(self):
        self.load_environment()
        self.tools = self.initialize_tools()
        self.agents = []
        self.tasks = []
        self.crew = None

    def load_environment(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

    def initialize_tools(self) -> Dict[str, Any]:
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
        if tools is None:
            tools = []
        agent = Agent(
            role=role, goal=goal, backstory=backstory, verbose=verbose, tools=tools
        )
        self.agents.append(agent)
        return agent

    def create_task(self, name: str, description: str, tools: List[Any] = None) -> Task:
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
        if self.crew is None:
            raise ValueError("Crew not created. Call create_crew() first.")
        return self.crew.kickoff()

    def get_user_input(self) -> str:
        print("Welcome!")
        return input("What is your question? ")


def main():
    crew_manager = CrewManager()

    # Create agents
    crew_manager.create_agent(role="", goal="", backstory="", tools=[], verbose=False)

    crew_manager.create_agent(role="", goal="", backstory="", tools=[], verbose=False)

    crew_manager.create_agent(role="", goal="", backstory="", tools=[], verbose=False)

    # Create tasks
    crew_manager.create_task(name="", description="", tools=[])

    crew_manager.create_task(name="", description="", tools=[])

    # Create and run crew
    crew_manager.create_crew()
    question = crew_manager.get_user_input()
    crew_manager.run_crew()


if __name__ == "__main__":
    main()
