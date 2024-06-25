import os
from io import BytesIO
import streamlit as st
from docx import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew

search_tool = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="Search the web using DuckDuckGo. Give argument - {'query': '<Whatever you want to search>'}",
)

def generate_text(llm, topic, depth):
    inputs = {'topic': topic}
   
    manager = Agent(
        role='Debate Manager',
        goal='Ensure adherence to debate guidelines and format',
        backstory="""Experienced Debate Manager adept at overseeing structured debates across various domains...""",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    proponent = Agent(
        role='Proponent of Topic',
        goal="Present the most convincing arguments in favor of the topic, using factual evidence and persuasive rhetoric.",
        backstory="""You are an exceptional debater, having recently won the prestigious World Universities Debating Championship...""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    opposition = Agent(
        role='Opponent of Topic',
        goal="Present the most convincing arguments against the topic, using logical reasoning and ethical considerations.",
        backstory="""You are a distinguished debater, recognized for your analytical skills and ethical reasoning...""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    writer = Agent(
        role='Debate Summarizer',
        goal="Provide an unbiased summary of both sides' arguments...",
        backstory="""You are a highly respected journalist and author, known for your impartiality and clarity in reporting complex issues...""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task_manager = Task(
        description=f"Manage the debate flow according to the specified format...",
        agent=manager,
        expected_output="Successful management of the debate according to the task description"
    )

    task_proponent = Task(
        description=f'Research and Gather Evidence on {topic}.',
        agent=proponent,
        expected_output="A comprehensive list of compelling evidence, statistics, and expert opinions supporting the topic...",
        tools=[search_tool],
        context=[task_manager]
    )

    task_oppostion = Task(
        description=f'Research and Gather Evidence on {topic}',
        agent=opposition,
        expected_output="A comprehensive list of compelling evidence, statistics, and expert opinions opposing the topic...",
        tools=[search_tool],
        context=[task_manager]
    )

    task_writer = Task(
        description="Provide an unbiased summary of both sides' arguments...",
        agent=writer,
        expected_output="A well-structured debate transcript featuring opening statements, rebuttals, and closing statements...",
        context=[task_manager, task_proponent, task_oppostion]
    )

    crew = Crew(
        agents=[manager, proponent, opposition, writer],
        tasks=[task_manager, task_proponent, task_oppostion, task_writer],
        verbose=2,
        context={"topic": topic}
    )

    result = crew.kickoff(inputs=inputs)
    return result

def setup_openai(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(temperature=0.6, max_tokens=2000)

def setup_gemini(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.6,
        google_api_key=api_key
    )

def setup_groq(api_key):
    return ChatGroq(
        api_key=api_key,
        model='llama3-70b-8192'
    )

def main():
    st.header('Debate Generator')

    with st.sidebar:
        with st.form('Gemini/OpenAI/Groq'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI', 'Groq'))
            api_key = st.text_input(f'Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

    llm = None
    if api_key:
        if model == 'OpenAI':
            llm = setup_openai(api_key)
        elif model == 'Gemini':
            llm = setup_gemini(api_key)
        elif model == 'Groq':
            llm = setup_groq(api_key)

    topic = st.text_input("Enter the debate topic:")
    depth = st.text_input("Enter the depth required:")

    if st.button("Generate Content") and llm:
        with st.spinner("Generating content..."):
            try:
                generated_content = generate_text(llm, topic, depth)
                st.markdown(generated_content)

                doc = Document()
                doc.add_heading(topic, 0)
                doc.add_paragraph(generated_content)

                buffer = BytesIO()
                doc.save(buffer)
                buffer.seek(0)

                st.download_button(
                    label="Download as Word Document",
                    data=buffer,
                    file_name=f"{topic}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
