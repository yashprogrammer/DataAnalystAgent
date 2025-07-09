from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from dotenv import load_dotenv
from autogen_agentchat.base import TaskResult
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from typing import Sequence,List
import asyncio
import os
import pandas as pd



api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI model client
llm1 = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)


def get_titanic_dataset() -> pd.DataFrame:
    """Returns the Titanic dataset as a pandas DataFrame."""
    df = pd.read_csv('Titanic-Dataset.csv')
    return df

def count_total_passengers() -> int:
    """Returns the total number of passengers in the dataset."""
    df = pd.read_csv('Titanic-Dataset.csv')
    return len(df)

def count_survivors() -> int:
    """Returns the total number of survivors (Survived == 1)."""
    df = pd.read_csv('Titanic-Dataset.csv')
    return df['Survived'].sum()

def count_non_survivors() -> int:
    """Returns the total number of passengers who died (Survived == 0)."""
    df = pd.read_csv('Titanic-Dataset.csv')
    return len(df) - df['Survived'].sum()

def survival_rate() -> float:
    """Returns the percentage of passengers who survived."""
    df = pd.read_csv('Titanic-Dataset.csv')
    return (df['Survived'].sum() / len(df)) * 100

def class_distribution() -> pd.Series:
    """Returns the number of passengers in each passenger class (Pclass)."""
    df = pd.read_csv('Titanic-Dataset.csv')
    return df['Pclass'].value_counts()




Planner_agent = AssistantAgent(
    name='PlannerAgent',
    description='An agent for planning the data analysis, this agent should be first to engage when given a new task',
    model_client=llm1,
    system_message='''
    You are a Data analysis planning agent
    your job is to break down the  complex data analysis task into smaller, manageble subtasks.
    the complex tasks will be related to perform data analysis on titanic based on a Dataset with "Titanic-Dataset.csv"
    you team members are :
    DataAnalyserAgent: Perform data analysis and extract info from dataset using tools.
    UserAgent : Approve/Reject current flow by observing the current progress and provide feedback.
    CodeGeneratorAgent : Generate python and terminal code for execution at CodeExecutorAgent.
    CodeExecutorAgent: Perform python code execution to extract data from "Titanic-Dataset.csv" and generate relevant graphs.

    You only plan and delegate tasks - you do not execute yourself.

    When assisning the tasks, use the below format : 
    1. <agent> : <task>

    After all the tasks are completed, summarize the findings and print out 'TERMINATE'.
    Only say 'TERMINATE' to stop the conversion and never in other cases
    '''
)


DataAnalyserAgent = AssistantAgent(
    name='DataAnalyserAgent',
    description='An agent for performing data analysis and extracting info from dataset using tools',
    model_client=llm1,
    tools=[get_titanic_dataset, count_total_passengers,count_survivors,count_non_survivors,survival_rate,class_distribution],
    system_message='''
    You are a Data analyser agent.
    List of the tools you have -
    `get_titanic_dataset()` - Returns the Titanic dataset as a pandas DataFrame.
    `count_total_passengers()` - Returns the total number of passengers in the dataset.
    `count_survivors()` - Returns the total number of survivors (where Survived == 1).
    `count_non_survivors()` - Returns the total number of passengers who did not survive (Survived == 0).
    `survival_rate()` - Returns the percentage of passengers who survived.
    `class_distribution()` - Returns the number of passengers in each passenger class (`Pclass`).

    you only call only one tool at a time

    once you have the result from tool, you perform calcuation and analysis on them to complete the step
    '''
)

CodeGeneratorAgent = AssistantAgent(
    name='CodeGeneratorAgent',
    description='An agent to generate code and shell commands for CodeExecutorAgent',
    model_client=llm1,
    tools=[get_titanic_dataset],
    system_message='''
    You are a code generator agent that is expert in performing data analysis using python and it's libraries
    You will be working with CodeExecutorAgent to execute code
    your first step compulsulorily must use get_titanic_dataset to get df, then store the df as 'docker_titanic_ds.csv' and use this csv to perform all calculations
    You will be give a task and you should first provide a way to solve the task/problem
    Then you should give the code in Python Block format so that it can be ran by code executor agent
    You can provide Shell scipt as well if code fails due to missing libraries, make sure to use pip install command
    You should give the corrected code in Python Block format if error is there
    Once the code has been successfully executed and you have the results. You should pass the result to PlannerAgent
    if you have to save the file, save it with output.png or output.txt or output.gif
    '''
)

docker=DockerCommandLineCodeExecutor(
    image='jupyter/scipy-notebook:latest',
    work_dir='tmp',
    timeout=120
)

code_executor_agent = CodeExecutorAgent(
    name='CodeExecutorAgent',
    description="An agent that executes code in a Docker container.",
    code_executor=docker,
)




UserAgent = UserProxyAgent(
    name='UserAgent',
    description='A proxy agent that represent a user',
    input_func=input
)

combined_termination = TextMentionTermination('TERMINATE') | MaxMessageTermination(max_messages=15)


selector_prompt = '''
Select an agent to perform the task
{roles}

Current conversation history:
{history}

Read the above converstaion and then select an agent from {participants} to perform the next task,
Make sure that the PlannerAgent has assigned task before other agents start working.
only select one agent.

Never run CodeExecutorAgent without a code thread
'''


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if messages[-1].source == code_executor_agent.name:
        print('-'*50)
        print(f"The agent which we got last was {messages[-1].source}")
        return UserAgent.name
    return None

selector_team = SelectorGroupChat(
    participants=[Planner_agent,DataAnalyserAgent,UserAgent,CodeGeneratorAgent,code_executor_agent],
    model_client=llm1,
    termination_condition=combined_termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)



async def main():
    try:
        await docker.start()
        task='How many passengers boarded from each port and prepare a pie chart of it'
        stream = selector_team.run_stream(task=task)
        result = await Console(stream)
        print(result)

    except Exception as e:
        print(f"An error occured:{e}")

    finally:
        await docker.stop()

if(__name__=="__main__"):
    asyncio.run(main())
