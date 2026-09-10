import os

from dotenv import load_dotenv
from crewai import Agent, LLM, Task, Process,Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai.tools import tool

load_dotenv()
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


#Custom Tool
@tool("Find duplicate words")
def find_duplicates(text: str) -> list[str]:
    """Return words that appear more than once."""
    words = text.lower().split()
    duplicates = []
    for word in words:
        if words.count(word) > 1 and word not in duplicates:
            duplicates.append(word)
    return duplicates


# Current CrewAI talks to models through LiteLLM, not LangChain.
# Pass a crewai.LLM (or a model string). ChatGoogleGenerativeAI will not validate.
gemini = LLM(
    model="gemini/gemini-3.6-flash",
    temperature=0.5,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

#Researcher Agent
article_researcher = Agent(
    role="Senior Researcher",
    goal='Uncover ground breaking technologies in {topic}',
    backstory="Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world.",
    llm=gemini,
    verbose=True,
    memory=True,
    tools = [search_tool],
    allow_delegation=True
)

# Article Writer Agent using GPT
article_writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  llm=gemini,
  allow_delegation=False
)

#Duplicate Word Finder Agent
duplicate_finder = Agent(
  role='Duplicate Word Finder',
  goal='Find duplicate words in the finished article',
  verbose=True,
  memory=True,
  backstory=(
    "You take the writer's article, call the Find duplicate words tool on that text, "
    "and report the article plus the duplicates. You do not guess the list."
  ),
  tools=[find_duplicates],
  llm=gemini,
  allow_delegation=False
)

#Research Task
research_task = Task(
    description=(
        "Conduct a thorough analysis on the given {topic}."
        "Utilize SerperSearch for any necessary online research. "
        "Summarize key findings in a detailed report."
    ),
    expected_output='A detailed report on the data analysis with key insights.',
    tools=[search_tool],
    agent=article_researcher,
)


# Writing Task
writing_task = Task(
    description=(
        "Write an insightful article based on the data analysis report. "
        "The article should be clear, engaging, and easy to understand."
    ),
    expected_output='A 1-paragraph article summarizing the data insights.',
    agent=article_writer,
    tools=[search_tool],
    context=[research_task],
)

#Duplicate Finder Task
duplicate_finder_task = Task(
    description=(
        "Take the article from the previous task. "
        "Call the Find duplicate words tool on the full article. "
        "Return the original article, then the list the tool returned."
    ),
    expected_output=(
        "Two sections:\n"
        "Article: <the paragraph>\n"
        "Duplicates: <the list from the Find duplicate words tool>"
    ),
    agent=duplicate_finder,
    tools=[find_duplicates],
    context=[writing_task],
)


crew = Crew(
        agents = [article_researcher,article_writer,duplicate_finder],
        tasks = [research_task,writing_task,duplicate_finder_task],
        process = Process.sequential
)

research_inputs = {
                'topic' : 'The rise of British Empire'
}

result = crew.kickoff(inputs=research_inputs)

print(result)