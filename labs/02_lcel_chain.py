from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise technical explainer for senior engineers."),
        ("human", "Explain {topic} like I'm a senior backend engineer, in 3 bullet points."),
    ]
)

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
parser = StrOutputParser()

# LCEL: each component is a Runnable; `|` pipes output → input.
chain = prompt | model | parser

result = chain.invoke({"topic": "LangChain"})
print(result)
