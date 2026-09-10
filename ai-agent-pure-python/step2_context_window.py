from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GROQ_API_KEY")
)

messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ("exit","quit"):
        break
    
    messages.append({"role": "user","content": user_input})
    
    response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages
    )
    
    reply = response.choices[0].message.content
    messages.append({"role": "assistant","content": reply})
    print(messages)
    print("Bot:", reply)
    
