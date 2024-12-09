# import tghe modules for prompts, agents, tools

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain import hub
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent
from langchain.chains import LLMChain

from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools.python.tool import PythonREPLTool

from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferMemory

from langchain.prompts import MessagesPlaceholder


# 1. Cypher query crafting tool

def execute_cypher_query(cypher_query: str) -> str:
    """ Use this tool to execute Cypher queries on a Neo4j database."""
    print("I am in the execute_cypher_query function")
    return cypher_query


# def lookup(question: str)-> str:

def lookup(question: str, chat_history: List[Dict[str, Any]] = []):
    llm = ChatOpenAI(model_name='gpt-4-turbo-preview', temperature=0)

    # prepare the prompt template

    template = '''
        Answer the following questions as best as you can.
        use only the tools provided
        Questions: {q}
        {chat_history}


         Additional Information:

                Use this additional information for neo4j cypher query only.
                Based on the user's input, construct a single, valid Neo4j Cypher query.
                The schema of the Neo4j database is:

                (p:ProductionLine)-[:HAS_CARRIER]->(c:CarrierOrWorkcenter)-[:PERFORMS_ACTION]->(f:Function)-[:ASSOCIATED_WITH]-(a:ActivitySet)-[:HAS_ERROR]-(e:ErrorCode).

                Note: In the Errorcode nodes, there is no 'name' property.

                Always use the schema information to validate and construct queries.

                if you can't access the neo4j database just return the query.
                Remove the word cypher from your output 

        '''

    prompt_template = PromptTemplate.from_template(template)

    prompt = hub.pull('hwchase17/react')  # name of the react prompt

    # new one
    # rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # lets see the type
    # print(type(prompt))

    # lets see the input variables  of the  react prompt

    # print(prompt.input_variables)

    # to see template

    # print(prompt.template)

    # we will use three tools. The agent will chose the best tools based on  user's query

    cypher_query_tool = Tool(
        name='Cypher Query Executor',
        func=execute_cypher_query,
        description='Useful when you need to  excute cyper query'

    )

    # 2 Python REPL tool (for executing Python code)

    python_repl = PythonREPLTool()
    python_repl_tool = Tool(
        name='Python REPL',
        func=python_repl.run,
        description='Useful when you need to use Python to answer a question. You need input Python code'

    )

    # 3. Wikipedia Tool (for searching Wikipedia)

    api_wrapper = WikipediaAPIWrapper()
    wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)

    wikipedia_tool = Tool(
        name='Wikipedia',
        func=wikipedia.run,
        description='Useful for when you need to look up a topic, country , or person on Wikipedia. '

    )

    # 4. DuckDuckGo Search Tool (for general web searches)

    search = DuckDuckGoSearchRun()
    duckduckgo_tool = Tool(
        name='DuckDuckGoSearch',
        func=search.run,
        description='Useful for when you need to perform an internet search to find information that another tool can\'t provide'

    )

    # collect all the tools in a list

    tools = [cypher_query_tool, python_repl_tool, wikipedia_tool, duckduckgo_tool]

    # to create the agent there are two things
    # 1. load the tools that the agent will use
    # 2 initiatialise the agent using the tool executor

    # agent = create_react_agent(llm, tools, prompt)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # setup an agent executor to run the agent
    # The agent executor examines the input and determines the best tool to use for the user query
    # the agent executor also manages the agent input and output

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,

        # controls how the agent exexutor handles ptential passing errors.
        # If its sets to true it will gracefully hande parsing erors and continue with execution process
        # if set to false which is default, parsing errors will caus execution to stop raising excepton

        max_iterations=10
        # sets a max limit  on the number of steps or iterations the agent can take during a single execution
        # prevents agents from running indefinitely which can lead to performance issues

    )

    # Lets prepare a question to ask the agent
    # we will start with a programming question to see if the agent using the REPL tool

    # question = 'Generate the first 20 numbers in the fibonacci series'

    # question ="who is current PM of UK ?"

    # question = "Tell me all ErrorCode nodes with the code '2' "

    result = agent_executor.invoke({
        'input': prompt_template.format(q=question, chat_history=chat_history)
        # q is the placeholder in the prompttemplate . input is one ofthe input variables f the prompt

    })

    print("Interroagting the result dictionary")
    for key in result.keys():
        print(key)
        print(result[key])

    return result


if __name__ == "__main__":
    question = input("Please ask my agent a question")
    finalanswer = lookup(question=question)
    # print(finalanswer)


