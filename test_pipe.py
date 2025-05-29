from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=local_pipeline)

response = llm.invoke("What is the capital of France?")
print(response)
