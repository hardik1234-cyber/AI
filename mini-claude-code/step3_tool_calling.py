from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GROQ_API_KEY")
)


def read_file(path):
    try:
        with open(path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {path} not found!"
    
    
#Schemas is the format an LLM reads the tool and uses it , If we use any frameworks like Langgraph then it is auto generated.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters":{
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to read"},
                },
                "required": ["path"],
            },
        },
    },
]

messages = [
    {"role": "user", "content": "What is inside notes.txt? Summarize in one line."},
]

while True:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages, # type: ignore
        tools=TOOL_SCHEMAS, # type: ignore
    )

    message = response.choices[0].message
    messages.append(message) # type: ignore

    # No tool calls means the model is done and gave us a normal answer
    if not message.tool_calls:
        print(message.content)
        break

    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments) # type: ignore

        print(f"Model wants to run: read_file({args})")

        result = read_file(**args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })