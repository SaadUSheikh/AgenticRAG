import os
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# ✅ Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ✅ Define Agents for Hospital Appointment use case
appointment_agent = Agent(
    role="Appointment Scheduler",
    goal="Help users schedule hospital appointments based on their medical needs and timing.",
    backstory="You're a digital assistant trained to understand patient requirements and schedule them with the right department.",
    llm=llm
)

followup_agent = Agent(
    role="Follow-up Advisor",
    goal="Provide follow-up care details and estimated visit costs based on the department or treatment.",
    backstory="You're a medical assistant that helps patients understand the next steps and potential costs after scheduling an appointment.",
    llm=llm
)

# ✅ Define Tasks (with expected output + output format)
appointment_task = Task(
    description=(
        "The user request is: '{{ query }}'.\n"
        "Determine the most relevant hospital department based on symptoms.\n"
        "Then suggest an appropriate appointment time.\n"
        "Departments: Cardiology, Dermatology, Neurology, Orthopedics, General Medicine.\n"
        "You must reply in this exact format:\n"
        "Thought: <your reasoning>\nFinal Answer: <department> on <date/time>"
    ),
    expected_output="Scheduled department and appointment time.",
    output_format="Thought: <your reasoning>\nFinal Answer: <department> on <date/time>",
    agent=appointment_agent
)

followup_task = Task(
    description=(
        "You will receive the user's department selection from the previous step as '{{ query }}'.\n"
        "Use this department name to provide follow-up care suggestions and an estimated cost.\n"
        "If the department is not recognized, assume 'General Medicine'.\n"
        "You must reply in this exact format:\n"
        "Thought: <your reasoning>\nFinal Answer: <follow-up steps and cost estimate>"
    ),
    expected_output="Follow-up care details and cost estimate.",
    output_format="Thought: <your reasoning>\nFinal Answer: <follow-up steps and cost estimate>",
    agent=followup_agent
)

# ✅ Create Crew
crew = Crew(
    agents=[appointment_agent, followup_agent],
    tasks=[appointment_task, followup_task],
    verbose=True
)

# ✅ Run the crew with user input
query_input = "I have chest pain and need to see a doctor soon."
result = crew.kickoff(inputs={"query": query_input})
print("\n✅ Final Result:\n", result)
