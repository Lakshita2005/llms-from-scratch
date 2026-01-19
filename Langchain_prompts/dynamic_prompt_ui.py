from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text2text-generation"
)
model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_title = st.text_input("Enter the title of the research paper: ")
explanation_style = st.text_input("Enter the explanation style (e.g., simple, detailed, technical): ")
explanation_length = st.text_input("Enter the explanation length (e.g., short, medium, long): ")

template = load_prompt('template.json')

# template 
# template = PromptTemplate(
#     template="""
#     Please summarize the research paper titled "{paper_title}" with the following specifications:
#     Explanation Style: {explanation_style}
#     Explanation Length: {explanation_length}
#     1. Mathematical Details: 
#         - Include relevant equations and derivations if present in the paper.
#         - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
#     2. Analogies:
#         - Use relatable analogies to clarify complex ideas.
#     If certain information is not available in the paper, respond with "Insufficient information".
#     Ensure the summary is clear , accurate and aligns with the specified style and length.
#     """,
#     input_variables=["paper_title", "explanation_style", "explanation_length"],
#     validate_template=True,
# )

# fill the placeholders
prompt = template.invoke({
    'paper_title':paper_title,
    'explanation_style':explanation_style,
    'explanation_length': explanation_length
})
if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)