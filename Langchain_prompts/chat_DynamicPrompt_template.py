from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}')
])

user_domain = input('Enter your domain: ')
user_topic = input('Enter the topic you want to know about: ')

prompt = chat_template.invoke({'domain':user_domain,'topic':user_topic})

result = model.invoke(prompt)
print(result.content)