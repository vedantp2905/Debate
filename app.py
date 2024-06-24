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
   
    manager = Agent(
    role='Debate Manager',
    goal='Ensure adherence to debate guidelines and format',
    backstory="""Experienced Debate Manager adept at overseeing structured debates
    across various domains. Skilled in maintaining decorum, managing time efficiently,
    and resolving unforeseen issues. Decisive and calm under pressure, ensuring
    successful and engaging debates.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

    proponent = Agent(
    role='Proponent of Topic',
    goal="""Present the most convincing arguments in favor of the topic,
    using factual evidence and persuasive rhetoric.""",
    backstory="""You are an exceptional debater, having recently won the prestigious
    World Universities Debating Championship. Your expertise lies in constructing
    compelling arguments that strongly support your stance on any given topic.
    You possess a keen ability to present facts persuasively, ensuring your
    points are both convincing and grounded in reality.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
    )

    opposition = Agent(
    role='Opponent of Topic',
    goal="""Present the most convincing arguments against the topic,
    using logical reasoning and ethical considerations.""",
    backstory="""You are a distinguished debater, recognized for your analytical
    skills and ethical reasoning. Recently, you were a finalist in the World
    Universities Debating Championship. Your strength lies in deconstructing
    arguments and highlighting potential flaws and ethical issues. You excel at
    presenting well-rounded and thoughtful counterarguments.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
    )

    writer = Agent(
    role='Debate Summarizer',
    goal="""Provide an unbiased summary of both sides' arguments. Synthesize the key points,
    evidence, and rhetorical strategies used by each side into a cohesive
    report that helps the audience understand the full scope of the debate.""",
    backstory="""You are a highly respected journalist and author, known for
    your impartiality and clarity in reporting complex issues. You have
    moderated presidential debates and written award-winning articles that
    break down intricate topics for the general public. Your reputation is
    built on your ability to remain neutral, to synthesize diverse viewpoints,
    and to articulate complex arguments in a way that's accessible to all.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
    )

    task_manager = Task(
    description=f"""Manage the debate flow according to the specified format:
                   2- Both the debaters must present short concise opening statements starting with the proponent
                   3- The debaters must rebuttal based on the output of their opponent starting with the proponent 
                   4- The total rebuttal rounds should be equal to the: {depth}
                   5- The first rebuttal round should be based on opening statements of the debaters.
                   6- Each subsequent rebuttal round must build on the previous rebuttal round and dwelve deeper into the points presented in the previous rebuttal round.
                   7- Each debater must give a short and concise closing argument""",
    agent=manager,
    expected_output="Successful management of the debate according to the task description"
)

    task_proponent = Task(
        description=f'Research and Gather Evidence on {topic}',
        agent=proponent,
        expected_output="""A comprehensive list of compelling evidence, statistics,
        and expert opinions supporting the topic, sourced from reputable and
        credible publications and experts.""",
        tools=[search_tool],
        context=[task_manager]
    )

    task_oppostion = Task(
        description=f'Research and Gather Evidence on {topic}',
        agent=opposition,
        expected_output="""A comprehensive list of compelling evidence,
        statistics, and expert opinions opposing the topic, sourced from
        reputable and credible publications and experts.""",
        tools=[search_tool],
        context=[task_manager]  # Pass the task_manager as context
    )

    task_writer = Task(
        description="""Provide an unbiased summary of both sides'
        arguments, synthesizing key points, evidence, and rhetorical strategies
        into a cohesive report.""",
        agent=writer,
        expected_output="""A well-structured debate transcript featuring opening
        statements, rebuttals, and closing statements from both sides, followed by an impartial summary that
        captures the essence of the debate.
        Follow the format given to the debate manager""",
        context = [task_manager,task_proponent,task_oppostion]
    )

    crew = Crew(
        agents=[manager, proponent, opposition, writer],
        tasks=[task_manager, task_proponent, task_oppostion, task_writer],
        verbose=2,
        context={"topic": topic}
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

                content_lines = generated_content.split('\n')
                first_line = content_lines[0]
                remaining_content = '\n'.join(content_lines[1:])

                st.markdown(first_line)
                st.markdown(remaining_content)


                doc = Document()

                # Option to download content as a Word document
                doc.add_heading(topic, 0)
                doc.add_paragraph(first_line)
                doc.add_paragraph(remaining_content)

                buffer = BytesIO()
                doc.save(buffer)
                buffer.seek(0)

                st.download_button(
                    label="Download as Word Document",
                    data=buffer,
                    file_name=f"{topic}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
if __name__ == "__main__":
    main()
