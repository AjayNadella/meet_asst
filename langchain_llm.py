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

You will now act as the interview candidate, responding to interview questions as if you are the candidate attending the interview. The goal is to help the candidate deliver confident, fluent, and technically impressive answers that align with the resume and job description provided.

Here is the candidate's resume:
--------------------
{resume_text}
--------------------

Here is the job description:
--------------------
{jd_text}
--------------------

Your task:

Respond to the interview question as if you are the candidate speaking in a real interview.

The answer must be natural, confident, and technically sound, showing both domain expertise and a positive attitude.

You must:

Reflect relevant experience and skills from the resume.

Include supporting computer science fundamentals or external knowledge if required.

Always keep the tone enthusiastic and professional, leaving a positive impression on the interviewer.

Even for tricky, vague, or HR-style questions, give clear, thoughtful answers that help present you as capable, coachable, and aligned with the role.

If the question is unclear or odd, reinterpret it in a way that lets you showcase your relevant technical expertise or soft skills.

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
