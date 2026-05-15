import argparse
import json
import os
import subprocess
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                },
            },
        },
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        choice = chat.choices[0]
        calls = choice.message.tool_calls
        messages.append(choice.message)

        if not calls:
            print(choice.message.content)
            break

        for call in calls:
            if call.function.name == "Read":
                args = json.loads(call.function.arguments)
                with open(args["file_path"]) as f:
                    result = f.read()
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            elif call.function.name == "Write":
                args = json.loads(call.function.arguments)
                with open(args["file_path"], "w") as f:
                    f.write(args["content"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Successfully wrote to file",
                    }
                )
            elif call.function.name == "Bash":
                args = json.loads(call.function.arguments)
                os.system(args['command'])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Successfully executed a command",
                    }
                )


if __name__ == "__main__":
    main()
