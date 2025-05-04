# langchain_llm.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from resume_loader import load_resume_and_jd
resume_text, jd_text = load_resume_and_jd()


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")



# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["input"],
    template=f"""
You are an AI interview assistant specialized in Machine Learning, Deep Learning, NLP,GEN AI,end to end deployment.

Here is the candidate's resume:
--------------------
{resume_text}
--------------------

Here is the job description:
--------------------
{jd_text}
--------------------

Answer the following interview question as if you are helping the candidate.
provide clear, concise answers based on resume and job description and other external knowledge if they asked more questions related to ML/AI/GEN AI.

Question: {{input}}
Answer:"""
)

chain = prompt_template | ChatGroq(
        temperature=0.5,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )



def get_answer(input_text: str) -> str:
    """Runs the prompt with user input and returns LLaMA's response."""
    try:
        response = chain.invoke(input=input_text)
        
        if hasattr(response, 'content'):
             return response.content.strip()
        
        return str(response).strip()
    except Exception as e:
        print("❌ LLM error:", e)
        return "Error getting response."
