import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# ✅ Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# ✅ Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ✅ Define Agents (no tools)
destination_agent = Agent(
    role="Travel Recommender",
    goal="Suggest a travel destination based on the user's interests.",
    backstory="You're a seasoned travel advisor helping people pick great destinations.",
    llm=llm
)

budget_agent = Agent(
    role="Budget Advisor",
    goal="Estimate a travel budget based on selected destination.",
    backstory="You provide smart, personalized cost estimates for trips around the world.",
    llm=llm
)

# ✅ Define Tasks (with expected output + output format)
destination_task = Task(
    description=(
        "The user said: '{{ query }}'.\n"
        "Based on this, recommend a travel destination from:\n"
        "- Beach → Maldives\n"
        "- Mountain → Swiss Alps\n"
        "- Adventure → New Zealand\n"
        "- History → Rome\n"
        "Else, recommend Paris."
    ),
    expected_output="A single recommended destination.",
    output_format="Thought: <your reasoning>\nFinal Answer: <your destination>",
    agent=destination_agent
)

budget_task = Task(
    description=(
        "Use the destination selected in the previous step to estimate the travel budget.\n"
        "Destination-based budget:\n"
        "- Maldives: $3000\n"
        "- Swiss Alps: $3500\n"
        "- New Zealand: $4000\n"
        "- Rome: $2500\n"
        "- Paris: $2700"
    ),
    expected_output="Estimated travel cost in USD.",
    output_format="Thought: <your reasoning>\nFinal Answer: <your estimated budget>",
    agent=budget_agent
)

# ✅ Create and Run the Crew
crew = Crew(
    agents=[destination_agent, budget_agent],
    tasks=[destination_task, budget_task],
    verbose=True
)

# ✅ Run it
result = crew.kickoff(inputs={"query": "I love mountain hiking and adventure."})
print(result)
