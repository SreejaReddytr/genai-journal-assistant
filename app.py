import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
import os

# --- Custom Styles ---
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(to bottom, #fdfbfb, #ebedee);
        color: #222;
    }
    .stApp {
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .subheading {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 20px;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stTabs [role="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- OpenAI Key Setup ---
openai_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("🔐 Enter OpenAI API Key:", type="password")
if not openai_key:
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# --- Prompt Chains ---
learning_prompt = PromptTemplate(input_variables=["role"], template="Create a 6-week roadmap to become a {role} with study materials and links.")
learning_chain = LLMChain(llm=llm, prompt=learning_prompt)
def generate_learning_pathway(role: str, **kwargs): return learning_chain.run(role=role)

job_prompt = PromptTemplate(input_variables=["role"], template="List 5 entry-level jobs for a {role} with apply links.")
job_chain = LLMChain(llm=llm, prompt=job_prompt)
def search_open_roles(role: str, **kwargs): return job_chain.run(role=role)

meal_prompt = PromptTemplate(input_variables=["veggies", "goal"], template="Give a 7-day South Indian meal plan using {veggies} for a {goal} lifestyle. Include recipe links.")
meal_chain = LLMChain(llm=llm, prompt=meal_prompt)
def generate_meal_plan(veggies: str = "spinach, tomato", goal: str = "balanced", **kwargs): return meal_chain.run(veggies=veggies, goal=goal)

affirm_prompt = PromptTemplate(input_variables=["emotion"], template="Give a motivational affirmation for someone feeling {emotion}.")
affirm_chain = LLMChain(llm=llm, prompt=affirm_prompt)
def generate_affirmation(emotion: str, **kwargs): return affirm_chain.run(emotion=emotion)

# --- Tools + Agent ---
tools = [
    Tool(name="generate_learning_pathway", func=generate_learning_pathway, description="Create a roadmap for a career goal."),
    Tool(name="search_open_roles", func=search_open_roles, description="Find jobs with links."),
    Tool(name="generate_meal_plan", func=generate_meal_plan, description="Suggest a healthy South Indian meal plan."),
    Tool(name="generate_affirmation", func=generate_affirmation, description="Give a motivational affirmation.")
]
agent_executor = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)

# --- Header ---
st.markdown("<h1>GenAI Journal Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheading'>Helping you navigate your career, health, and emotions using the power of GenAI 🚀</div>", unsafe_allow_html=True)

# --- Tabs for Feature Selection ---
tabs = st.tabs(["📚 Study Roadmap", "💼 Job Search", "🍛 Meal Plan", "💬 Affirmation"])

with tabs[0]:
    role = st.text_input("🎓 What role do you want to become?")
    if st.button("Generate Study Plan"):
        if role:
            with st.spinner("Creating roadmap..."):
                output = generate_learning_pathway(role)
                st.markdown(output)
        else:
            st.warning("Please enter a role.")

with tabs[1]:
    job_role = st.text_input("💼 Search jobs for what role?")
    if st.button("Find Jobs"):
        if job_role:
            with st.spinner("Searching jobs..."):
                output = search_open_roles(job_role)
                st.markdown(output)
        else:
            st.warning("Please enter a job title.")

with tabs[2]:
    veggies = st.text_input("🥦 Favorite veggies (comma-separated):", value="spinach, tomato")
    goal = st.selectbox("🎯 Goal", ["balanced", "muscle gain", "weight loss", "diabetic-friendly"])
    if st.button("Get Meal Plan"):
        with st.spinner("Generating meal plan..."):
            output = generate_meal_plan(veggies=veggies, goal=goal)
            st.markdown(output)

with tabs[3]:
    emotion = st.text_input("😔 How are you feeling?")
    if st.button("Get Affirmation"):
        if emotion:
            with st.spinner("Finding motivation..."):
                output = generate_affirmation(emotion)
                st.markdown(output)
        else:
            st.warning("Please enter an emotion.")
