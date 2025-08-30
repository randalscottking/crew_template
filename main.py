import os

from crewai import Agent, Crew, Process, Task
from crewai_tools import (
    CSVSearchTool,
    DirectoryReadTool,
    FileReadTool,
    RagTool,
    SerperDevTool,
)
from dotenv import load_dotenv

# Use .env file for llm credentials
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

# Define Tools (https://docs.crewai.com/en/concepts/tools#available-crewai-tools)
docs_tool = DirectoryReadTool(directory="./blog-posts")
file_tool = FileReadTool()
search_tool = SerperDevTool()
csv_search_tool = CSVSearchTool()
rag_tool = RagTool()

# Create Agents
agent1 = Agent(role="", goal="", backstory="", verbose=False, tools=[])

agent2 = Agent(role="", goal="", backstory="", verbose=False, tools=[])

agent3 = Agent(role="", goal="", backstory="", verbose=False, tools=[])

# Create Tasks
task1 = Task(name="", description="", tools=[])

task2 = Task(name="", description="", tools=[])

# Create Crew
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2],
    verbose=True,
    planning=True,
    process=Process.hierarchical,
    manager_llm="gpt-4o",
)

# UI
print("Welcome!")
question = input("What is your question? ")

# Run the Tasks
crew.kickoff()
