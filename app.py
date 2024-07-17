from io import BytesIO
import re
import os
import asyncio
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
import pandas as pd
from langchain_community.tools import DuckDuckGoSearchResults

def verify_gemini_api_key(api_key):
    API_VERSION = 'v1'
    api_url = f"https://generativelanguage.googleapis.com/{API_VERSION}/models?key={api_key}"
    
    try:
        response = requests.get(api_url, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # If we get here, it means the request was successful
        return True
    
    except requests.exceptions.HTTPError as e:
        
        return False
    
    except requests.exceptions.RequestException as e:
        # For any other request-related exceptions
        raise ValueError(f"An error occurred: {str(e)}")

def verify_gpt_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Using a simple request to the models endpoint
    response = requests.get("https://api.openai.com/v1/models", headers=headers)
    
    if response.status_code == 200:
        return True
    elif response.status_code == 401:
        return False
    else:
        print(f"Unexpected status code: {response.status_code}")
        return False
    
def verify_groq_api_key(api_key):
    api_url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # If we get here, it means the request was successful
        return True
    
    except requests.exceptions.HTTPError as e:
        
        return False
    
    except requests.exceptions.RequestException as e:
        # For any other request-related exceptions
        raise ValueError(f"An error occurred: {str(e)}")

search_tool = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="""Search the web using DuckDuckGo. Give argument -
                {"query": "<Whatever you want to search>"}""",
)

def set_column_width_and_wrap(workbook, worksheet, df):
    for idx, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
        cell_format = workbook.add_format({'text_wrap': True})
        worksheet.set_column(idx, idx, max_len, cell_format)

def generate_text(llm, topic, depth):
    inputs = {'topic': topic}
   
    manager = Agent(
        role='Debate Manager',
        goal='Ensure adherence to debate guidelines and format',
        backstory="""Experienced Debate Manager adept at overseeing structured debates
        across various domains. Skilled in maintaining decorum, managing time efficiently,
        and resolving unforeseen issues. Decisive and calm under pressure, ensuring
        successful and engaging debates.""",
        verbose=True,
        allow_delegation=False,
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
                       6- Each subsequent rebuttal round must build on the previous rebuttal round and delve deeper into the points presented in the previous rebuttal round.
                       7- Each debater must give a short and concise closing argument""",
        agent=manager,
        expected_output="Successful management of the debate according to the task description"
    )

    task_proponent = Task(
        description=f'''Research and Gather Evidence on {topic}.''',
        agent=proponent,
        expected_output="""A comprehensive list of compelling evidence, statistics,
        and expert opinions supporting the topic, sourced from reputable and
        credible publications and experts.""",
        tools=[search_tool],
        context=[task_manager]
    )

    task_opposition = Task(
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
        context=[task_manager, task_proponent, task_opposition]
    )

    crew = Crew(
        agents=[manager, proponent, opposition, writer],
        tasks=[task_manager, task_proponent, task_opposition, task_writer],
        verbose=2,
        context={"topic": topic}
    )

    result = crew.kickoff(inputs=inputs)
    return result

def main():
    st.header('Debate Generator')
    validity_model= False

    # Initialize session state
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = None
    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    if 'depth' not in st.session_state:
        st.session_state.depth = ""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'excel_buffer' not in st.session_state:
        st.session_state.excel_buffer = None
        
    with st.sidebar:
        with st.form('Gemini/OpenAI/Groq'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI', 'Groq'))
            api_key = st.text_input(f'Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

        if api_key:
            if model == "Gemini":
                validity_model = verify_gemini_api_key(api_key)
                if validity_model ==True:
                    st.write(f"Valid {model} API key")
                else:
                    st.write(f"Invalid {model} API key")
            elif model == "OpenAI":
                validity_model = verify_gpt_api_key(api_key)
                if validity_model ==True:
                    st.write(f"Valid {model} API key")
                else:
                    st.write(f"Invalid {model} API key")            
            elif model == "Groq":
                validity_model = verify_groq_api_key(api_key)
                if validity_model ==True:
                    st.write(f"Valid {model} API key")
                else:
                    st.write(f"Invalid {model} API key")
                
    if validity_model==True:
        if model == 'OpenAI':
            async def setup_OpenAI():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                os.environ["OPENAI_API_KEY"] = api_key
                llm = ChatOpenAI(model='gpt-4-turbo', temperature=0.6, max_tokens=2000, api_key=api_key)
                print("OpenAI Configured")
                return llm

            llm = asyncio.run(setup_OpenAI())

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

        elif model == 'Groq':
            async def setup_groq():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm = ChatGroq(
                    api_key=api_key,
                    model='llama3-70b-8192'
                )
                return llm

            llm = asyncio.run(setup_groq())

        topic = st.text_input("Enter the debate topic:", value=st.session_state.topic)
        depth = st.text_input("Enter the depth required:", value=st.session_state.depth)
        st.session_state.topic = topic
        st.session_state.depth = depth

        if st.button("Generate Content"):
            with st.spinner("Generating content..."):
                st.session_state.generated_content = generate_text(llm, topic, depth)
                process_content()

    if st.session_state.generated_content:

        st.markdown(st.session_state.generated_content)

        # Create a download button for Excel file if the buffer exists
        if st.session_state.excel_buffer is not None:
            st.download_button(
                label="Download as Excel",
                data=st.session_state.excel_buffer,
                file_name=f"{st.session_state.topic}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def process_content():
    # Process the generated content
    sections = re.split(r'\*\*(.*?):\*\*', st.session_state.generated_content)
    proponent_lines = []
    opponent_lines = []

    for i in range(1, len(sections), 2):
        speaker = sections[i].strip()
        content = sections[i+1].strip()
        
        if "Proponent" in speaker:
            proponent_lines.append(content)
            if len(opponent_lines) < len(proponent_lines):
                opponent_lines.append("")
        elif "Opponent" in speaker:
            opponent_lines.append(content)
            if len(proponent_lines) < len(opponent_lines):
                proponent_lines.append("")

    # Ensure both lists have the same length
    max_length = max(len(proponent_lines), len(opponent_lines))
    proponent_lines += [""] * (max_length - len(proponent_lines))
    opponent_lines += [""] * (max_length - len(opponent_lines))

    # Create a DataFrame and store in session state
    data = {'Proponent': proponent_lines, 'Opponent': opponent_lines}
    st.session_state.df = pd.DataFrame(data)

    # Create Excel file
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        st.session_state.df.to_excel(writer, index=False, sheet_name='Debate')
        workbook = writer.book
        worksheet = writer.sheets['Debate']
        
        # Adjust column widths and wrap text
        for idx, col in enumerate(st.session_state.df.columns):
            series = st.session_state.df[col].dropna()
            max_len = max((
                series.astype(str).map(len).max(),  # max length of values
                len(col)  # length of column name
            )) + 2  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)
        
        # Add text wrapping
        wrap_format = workbook.add_format({'text_wrap': True})
        worksheet.set_column('A:B', None, wrap_format)

    # Rewind the buffer and store in session state
    excel_buffer.seek(0)
    st.session_state.excel_buffer = excel_buffer.getvalue()

if __name__ == "__main__":
    main()
