# Requires: ollama serve + ollama pull llama3.2
from langchain_ollama import ChatOllama

# No API key needed — Ollama runs locally on http://localhost:11434
model = ChatOllama(model="llama3.2")

response = model.invoke("Explain LangChain in 2 sentences.")
print(response.content)
