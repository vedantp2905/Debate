from io import BytesIO
import os
import asyncio
from docx import Document
from langchain_groq import ChatGroq
from openai import OpenAI
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun(
        name="duckduckgo_search",
        description="""Search the web using DuckDuckGo. Give argument -
                    {"query": "<Whatever you want to search>"}""",
    )

def generate_text(llm, topic,depth):

    inputs = {'topic': topic}
   
    pro_topic = Agent(
        role='Proponent of Topic',
        goal=f"""Present the most convincing arguments in favor of the topic: {topic},
        using factual evidence and persuasive rhetoric.Make sure to conterargument opponent""",
        backstory="""You are an exceptional debater, having recently won the prestigious
        World Universities Debating Championship. Your expertise lies in constructing
        compelling arguments that strongly support your stance on any given topic.
        You possess a keen ability to present facts persuasively, ensuring your
        points are both convincing and grounded in reality.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    con_topic = Agent(
        role='Opponent of Topic',
        goal="""Present the most convincing arguments against the topic,
        using logical reasoning and ethical considerations. Make sure to conterargument opponent""",
        backstory="""You are a distinguished debater, recognized for your analytical
        skills and ethical reasoning. Recently, you were a finalist in the World
        Universities Debating Championship. Your strength lies in deconstructing
        arguments and highlighting potential flaws and ethical issues. You excel at
        presenting well-rounded and thoughtful counterarguments.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    
    task_pro = Task(
    description=f'''Research and Gather Evidence on {topic} using the provided search tool.
                    Using the evidence, write a concise argument supporting the topic taking into 
                    consideration the opposition's argument.''',
    agent=pro_topic,
    expected_output="""A well-researched and structured argument supporting the topic, 
                        addressing potential counterarguments and providing evidence to 
                        strengthen the pro stance. The argument should be logically 
                        coherent and persuasive, showcasing a deep understanding of the 
                        topic and its nuances.""",
    tools=[search_tool]
)


    task_con = Task(
        description=(f'''Research and Gather Evidence on {topic} using search tool given to you.
                     Using the Evidence write a concise argument against the topic taking into 
                     consideration opposition's argument'''),
        agent=con_topic,
        expected_output="""A well-researched and structured argument against the topic, 
                        addressing potential counterarguments and providing evidence to 
                        strengthen the pro stance. The argument should be logically 
                        coherent and persuasive, showcasing a deep understanding of the 
                        topic and its nuances.""",
        tools=[search_tool]
    )


    crew = Crew(
    agents=[pro_topic, con_topic],
    tasks=[task_pro, task_con],
    verbose=2
    
)

    result = crew.kickoff(inputs=inputs)
    return result

def main():
    
    st.header('Debate Generator')
    mod = None
        
    with st.sidebar:
        with st.form('Gemini/OpenAI/Groq'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI','Groq'))
            api_key = st.text_input(f'Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

    if api_key:
        if model == 'OpenAI':
            async def setup_OpenAI():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                os.environ["OPENAI_API_KEY"] = api_key
                llm = ChatOpenAI(temperature=0.6, max_tokens=2000)
                print("OpenAI Configured")
                return llm

            llm = asyncio.run(setup_OpenAI())
            mod = 'Gemini'


        elif model == 'Gemini':
            async def setup_gemini():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.6,
                    google_api_key=api_key
                )
                print("Gemini Configured")
                return llm

            llm = asyncio.run(setup_gemini())
            mod = 'Gemini'

            
        elif model == 'Groq':
            async def setup_groq():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm = ChatGroq(
                    api_key = api_key,
                    model = 'llama3-70b-8192'
                )
                return llm

            llm = asyncio.run(setup_groq())
            mod = 'Groq'

        topic = st.text_input("Enter the debate topic:")
        depth = st.text_input("Enter the depth required:")


        if st.button("Generate Content"):
            with st.spinner("Generating content..."):
                generated_content = generate_text(llm, topic, depth)

                st.markdown(generated_content)
                
if __name__ == "__main__":
    main()
