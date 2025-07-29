
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
import os

openai_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("🔐 Enter OpenAI API Key:", type="password")
if not openai_key:
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Learning Chain
learning_prompt = PromptTemplate(input_variables=["role"], template="Create a 6-week roadmap to become a {role} with study materials and links.")
learning_chain = LLMChain(llm=llm, prompt=learning_prompt)
def generate_learning_pathway(role: str, **kwargs): return learning_chain.run(role=role)

# Job Chain
job_prompt = PromptTemplate(input_variables=["role"], template="List 5 entry-level jobs for a {role} with apply links.")
job_chain = LLMChain(llm=llm, prompt=job_prompt)
def search_open_roles(role: str, **kwargs): return job_chain.run(role=role)

# Meal Chain
meal_prompt = PromptTemplate(input_variables=["veggies", "goal"], template="Give a 7-day South Indian meal plan using {veggies} for a {goal} lifestyle. Include recipe links.")
meal_chain = LLMChain(llm=llm, prompt=meal_prompt)
def generate_meal_plan(veggies: str = "spinach, tomato", goal: str = "balanced", **kwargs): return meal_chain.run(veggies=veggies, goal=goal)

# Affirmation Chain
affirm_prompt = PromptTemplate(input_variables=["emotion"], template="Give a motivational affirmation for someone feeling {emotion}.")
affirm_chain = LLMChain(llm=llm, prompt=affirm_prompt)
def generate_affirmation(emotion: str, **kwargs): return affirm_chain.run(emotion=emotion)

tools = [
    Tool(name="generate_learning_pathway", func=generate_learning_pathway, description="Create a roadmap for a career goal."),
    Tool(name="search_open_roles", func=search_open_roles, description="Find jobs with links."),
    Tool(name="generate_meal_plan", func=generate_meal_plan, description="Suggest a healthy South Indian meal plan."),
    Tool(name="generate_affirmation", func=generate_affirmation, description="Give a motivational affirmation.")
]

agent_executor = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)

st.title("🧠 GenAI Journal Assistant")
query = st.text_area("What's on your mind today?", height=200)

if st.button("Ask AI"):
    if query:
        with st.spinner("Thinking..."):
            result = agent_executor.invoke({"input": query, "chat_history": []})
            st.markdown(result["output"])
    else:
        st.warning("Please write something.")
