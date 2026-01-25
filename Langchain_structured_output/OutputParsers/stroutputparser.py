from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task='text_generation'
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed prompt
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line sumary on the following text: \n {text}',
    input_variables=['text']
)

# Instead of doing this we can use chains and create a pipeline for this
# prompt1 = template1.invoke({'topic':'black hole'})
# result = model.invoke(prompt1)
# prompt2 = template2.invoke({'text': result.content})
# final_result = model.invoke(prompt2)

# print(final_result.content)

parser = StrOutputParser()

# making chain because strOutputParser works better with chains
chain = template1 | model | parser |template2 | model | parser
# here parser is converting llm output into strings so it can be further used by llms

result = chain.invoke({'topic':'black hole'})

print("\nSummary: \n", result)