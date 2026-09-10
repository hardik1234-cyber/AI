from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
model="openai/gpt-oss-20b",
messages=[
    {"role": "system", "content": "You are a helpful AI assistant. Answer concisely in short sentences."},
    {"role": "user", "content": "Explain what an AI agent is in one sentence."},
    {"role": "assistant", "content": "An AI agent is simply an LLM connected to tools in a loop."},
    ],
)
print(response.choices[0].message.content)