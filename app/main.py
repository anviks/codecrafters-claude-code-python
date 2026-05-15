import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


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
            tools=[
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
                }
            ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        choice = chat.choices[0]
        messages.append(choice.message)

        if choice.message.content:
            print(choice.message.content)
            break
        else:
            calls = choice.message.tool_calls
            assert calls and len(calls) == 1
            call = calls[0]
            if call.function.name == "Read":
                args = json.loads(call.function.arguments)
                with open(args["file_path"]) as f:
                    result = f.read()
                print(result)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )


if __name__ == "__main__":
    main()
