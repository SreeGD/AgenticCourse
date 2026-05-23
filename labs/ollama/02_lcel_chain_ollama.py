# Requires: ollama serve + ollama pull llama3.2
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise technical explainer for senior engineers."),
        ("human", "Explain {topic} like I'm a senior backend engineer, in 3 bullet points."),
    ]
)

model = ChatOllama(model="llama3.2", temperature=0)
parser = StrOutputParser()

# LCEL: each component is a Runnable; `|` pipes output → input.
chain = prompt | model | parser

result = chain.invoke({"topic": "LangChain"})
print(result)
