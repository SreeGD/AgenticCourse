from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Swap to "claude-opus-4-7" for the most capable Claude model.
model = ChatAnthropic(model="claude-sonnet-4-6")

response = model.invoke("Explain LangChain in 2 sentences.")
print(response.content)
