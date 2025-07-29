import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType

# --- 🎨 UI Style ---
st.set_page_config(page_title="GenAI Journal Assistant", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f4f9fd;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ GenAI Journal Assistant")
st.subheader("Navigate your career, emotions, and health with the help of AI 💡")

# --- 🔐 API Key input ---
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

# --- 🔗 LLM and Chains ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)

learning_prompt = PromptTemplate(
    input_variables=["role"],
    template="Create a 6-week learning roadmap to become a {role} with study links."
)
learning_chain = LLMChain(llm=llm, prompt=learning_prompt)
def generate_learning_pathway(role): return learning_chain.run(role=role)

job_prompt = PromptTemplate(
    input_variables=["role"],
    template="List 5 entry-level jobs for a {role} with mock apply links."
)
job_chain = LLMChain(llm=llm, prompt=job_prompt)
def search_open_roles(role): return job_chain.run(role=role)

meal_prompt = PromptTemplate(
    input_variables=["veggies", "goal"],
    template="Create a 7-day South Indian meal plan using {veggies} for a {goal} lifestyle with 1 recipe link daily."
)
meal_chain = LLMChain(llm=llm, prompt=meal_prompt)
def generate_meal_plan(veggies, goal): return meal_chain.run(veggies=veggies, goal=goal)

affirm_prompt = PromptTemplate(
    input_variables=["emotion"],
    template="Give a short motivational affirmation for someone feeling {emotion}."
)
affirm_chain = LLMChain(llm=llm, prompt=affirm_prompt)
def generate_affirmation(emotion): return affirm_chain.run(emotion=emotion)

tools = [
    Tool(name="generate_learning_pathway", func=generate_learning_pathway, description="Career roadmap"),
    Tool(name="search_open_roles", func=search_open_roles, description="Find jobs"),
    Tool(name="generate_meal_plan", func=generate_meal_plan, description="7-day South Indian meal plan"),
    Tool(name="generate_affirmation", func=generate_affirmation, description="Motivational affirmation")
]

agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)

# --- 🧠 User Input ---
query = st.text_area("What’s on your mind today?", height=200)
if st.button("Ask AI"):
    if query:
        with st.spinner("Thinking..."):
            response = agent.invoke({"input": query})
            st.markdown(response["output"])
    else:
        st.error("Please enter a query.")


