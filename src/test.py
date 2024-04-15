import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

from langchain.llms import Ollama


import warnings
warnings.filterwarnings('ignore')

duckgo = DuckDuckGoSearchRun()
sematic = SemanticScholarQueryRun()
wiki = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

search_tools = [sematic, duckgo, wiki]

ollama_llm = Ollama(model="openhermes")

# Define your agents with roles and goals
team_lead = Agent(
  role='Experienced Team Leader for Researchers',
  goal='Oversee the preparation and process of paper publication creation for high-impact journals in the field of AI and aquaculture hybrid grouper',
  backstory="""As an experienced team leader for researchers in the specialized field of AI and aquaculture hybrid grouper, your responsibility is to guide the preparation 
  of research paper publications targeted at high-impact journals. When leading your team, you must follow this structured approach:

  ### Abstract:
  The study focuses on conducting thorough searches and reviews using specialized databases and relevant keywords to identify the latest research articles from 2020 to 2024 
  in the domain of AI and aquaculture hybrid grouper. Detailed summaries are created for each paper, highlighting the methodologies employed and the key outcomes achieved.

  ### Introduction:
  The introduction sets the stage for analyzing selected papers, with a specific focus on their methodologies, outcomes, and implications for AI applications in aquaculture, 
  particularly in hybrid grouper farming. The aim is to synthesize information from individual summaries to identify emerging trends and insights within the intersection of AI 
  and aquaculture hybrid grouper.

  ### Method:
  The methodology involves developing a comprehensive summary review that outlines the trends observed in the collected publications, emphasizing the advancements in AI technologies 
  for enhancing hybrid grouper aquaculture. A critical analysis of the methodologies employed is provided, along with the significance of the outcomes in improving aquaculture practices. 
  Potential directions for future research in AI and aquaculture hybrid grouper are also suggested.

  ### Discussion:
  The discussion delves into the implications of the analyzed papers, highlighting the advancements in AI technologies and their impact on hybrid grouper aquaculture. It explores the
  significance of the outcomes in improving aquaculture practices and discusses potential directions for further research in this field.

  ### Conclusion:
  In conclusion, the study emphasizes the importance of AI technologies in enhancing hybrid grouper aquaculture practices. It underscores the significance of the reviewed literature
  in advancing understanding and innovation in the field. The conclusion also sets the stage for future research and development in AI and aquaculture hybrid grouper.

  ### Reference:
  All included references are cited in accordance with academic standards to acknowledge the sources and provide credibility to the study.

  As the team leader, your role is pivotal in ensuring that the publication preparation process maintains high standards of quality and rigor, contributing to the dissemination of impactful research 
  findings in the specialized field of AI and aquaculture hybrid grouper.
  """,
  verbose=True,
  allow_delegation=True,
  tools=search_tools,
  llm=ollama_llm
)

researcher = Agent(
  role='Experienced Researcher',
  goal="""Gather at least 10 relevant research papers on the subject of AI and aquaculture hybrid grouper with the aim of sharing knowledge and insights to assist 
  in writing research papers for high-impact journals""",
  backstory="""As an experienced and meticulous researcher, your task involves efficiently summarizing abstracts, introductions, methods, discussions, and 
  conclusions from the gathered research papers to facilitate the transfer of valuable information to other agents.""",
  verbose=True,
  allow_delegation=False,
  llm=ollama_llm
)

writer = Agent(
  role='Highly Skilled Research Writer',
  goal="""Produce abstracts, introductions, methods, discussions, and conclusions for research papers focusing on AI and aquaculture hybrid grouper. 
  Collaborate with researcher agents to gather insights on the latest advancements in the field.""",
  backstory="""As a highly skilled research writer, your proficiency is in developing research papers with articulate abstracts, introductions, methods, 
  and discussions, simplifying intricate details for broader comprehension within the domain of AI and aquaculture hybrid grouper.""",
  verbose=True,
  allow_delegation=True,
  llm=ollama_llm
)


# Create tasks for your agents
task1 = Task(
  description="""Create a High-Impact Research Paper on the Cross-Functional Field of AI and Aquaculture Hybrid Grouper using the latest research paper from 2020 to 2024.

  Example:
  ### Objective:
  Develop a research paper for publication in a high-impact journal by systematically searching, reviewing, analyzing, and synthesizing relevant literature on the 
  intersection of Artificial Intelligence (AI) and aquaculture, specifically focusing on the hybrid grouper.

  ### Example Research Article:
  **Title:** Recent Advances in [Research Area]

  **Abstract:**
  This review paper provides a comprehensive overview of recent advancements in [research area], emphasizing key findings, methodologies, and trends from literature 
  published between 2020 and 2024. Through a systematic analysis of diverse research papers, the review aims to synthesize the current state of knowledge in [research area] 
  and identify emerging themes for future research.

  1. **Abstract:**
    - The abstract aims to identify relevant literature on AI applications in aquaculture, specifically focusing on the hybrid grouper. It outlines the steps involved in 
    defining search criteria, conducting a comprehensive search, and selecting high-quality studies that explore the cross-functional field of AI and aquaculture hybrid grouper.

  2. **Introduction:**
    - The introduction sets the objective of analyzing selected papers to extract insights on AI technologies in enhancing aquaculture practices, with a specific focus on the 
    hybrid grouper. It details the steps of extracting data from selected papers, synthesizing information to identify common themes and trends, and discussing the implications 
    of AI integration for advancing aquaculture practices, particularly in hybrid grouper farming.

  3. **Method:**
    - The methodology involves summarizing key findings related to AI's role in improving hybrid grouper farming efficiency and sustainability. It highlights the significance 
    of integrating AI technologies in aquaculture to enhance productivity and environmental sustainability. The method also discusses implications for researchers, practitioners, 
    and policymakers in leveraging AI advancements for sustainable aquaculture development.

  4. **Discussion:**
    - The discussion delves into the potential of AI in revolutionizing aquaculture practices for hybrid grouper production. It emphasizes the benefits of AI integration for 
    hybrid grouper cultivation, including efficiency improvements and sustainability enhancements. The discussion also explores collaboration opportunities and areas for further 
    research at the intersection of AI and aquaculture hybrid grouper production.

  5. **Conclusion:**
    - In conclusion, the study underscores the transformative potential of AI technologies in aquaculture, particularly for hybrid grouper farming. It highlights the positive 
    impact of AI integration on productivity, sustainability, and innovation in aquaculture practices. The conclusion emphasizes the need for continued research and collaboration 
    to harness the full benefits of AI in hybrid grouper production.

  6. **Reference:**
    - All sources and references cited in the study are included to acknowledge the contributions of existing literature and provide credibility to the research conducted on AI 
    and aquaculture hybrid grouper production.
  """,
  expected_output = """High impact research paper that contains abstract, introduction, method, discussion, conclusion and reference. The research paper should be based on the latest paper from
  2020 to 2024.
  """,
  output_file='output.txt',
  agent=team_lead
)

task2 = Task(
    description="""Write a review research paper tailored for submission to a high-impact journal focussing on AI application to hybrid grouper aquaculture.
    The paper should encompass essential sections including an introduction, current state of art method, research gap identification, comprehensive discussion, conclusion, and abstract. 
    Ensure each section is meticulously crafted to meet the standards of high-impact journals, showcasing the significance and novelty of the research findings.""",
    expected_output = "Full research paper with reference in markdown",
    agent=team_lead
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[team_lead, researcher, writer],
    tasks=[task1],
    verbose=2
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)