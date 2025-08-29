import os

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_community.tools import tool

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

print("Welcome!")
question = input("What is your question? ")

agent1 = Agent(role="", goal="", backstory="", verbose=False, tools=[])

agent2 = Agent(role="", goal="", backstory="", verbose=False, tools=[])

agent3 = Agent(role="", goal="", backstory="", verbose=False, tools=[])
