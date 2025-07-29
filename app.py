import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType

# --- 🎨 UI Config ---
st.set_page_config(page_title="GenAI Journal Assistant", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #F0F8FF;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("✨ GenAI Journal Assistant")
st.subheader("Navigate your career, emotions, and meal  with the help of AI 💡")

# --- 🔐 Use deployed secret API Key ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("🚨 Missing OpenAI API Key in Streamlit Cloud Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- 🔗 LLM Setup ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# --- 🛠 Prompt Chains ---
learning_prompt = PromptTemplate(
    input_variables=["role"],
    template="Create a 6-week learning roadmap to become a {role} with study links."
)
learning_chain = LLMChain(llm=llm, prompt=learning_prompt)

job_prompt = PromptTemplate(
    input_variables=["role"],
    template="List 5 entry-level jobs for a {role} with mock apply links."
)
job_chain = LLMChain(llm=llm, prompt=job_prompt)

meal_prompt = PromptTemplate(
    input_variables=["veggies", "goal"],
    template="Create a 7-day South Indian meal plan using {veggies} for a {goal} lifestyle with 1 recipe link daily."
)
meal_chain = LLMChain(llm=llm, prompt=meal_prompt)

affirm_prompt = PromptTemplate(
    input_variables=["emotion"],
    template="Give a short motivational affirmation for someone feeling {emotion}."
)
affirm_chain = LLMChain(llm=llm, prompt=affirm_prompt)

# --- 🧠 Tool Agent ---
tools = [
    Tool(name="generate_learning_pathway", func=lambda r: learning_chain.run(role=r), description="Career roadmap"),
    Tool(name="search_open_roles", func=lambda r: job_chain.run(role=r), description="Find jobs"),
    Tool(name="generate_meal_plan",func=lambda goal: meal_chain.run(veggies="mixed", goal=goal),description="Meal plan"),
    Tool(name="generate_affirmation", func=lambda e: affirm_chain.run(emotion=e), description="Motivational affirmation")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

# --- 📝 User Input ---
query = st.text_area("What’s on your mind today?", height=200)
if st.button("Ask AI"):
    if query.strip():
        with st.spinner("Thinking..."):
            response = agent.invoke({"input": query})
            st.markdown(response["output"])
    else:
        st.warning("Please enter a prompt.")
