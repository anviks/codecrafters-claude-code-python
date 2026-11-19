import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
SKILLS_PATH = Path(".claude/skills/")

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
    {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "Load a skill's instructions into the conversation",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to use",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments for the skill",
                    },
                },
            },
        },
    },
]


def apply_skill(skill: str, args: list[str]) -> str:
    skill_text = (SKILLS_PATH / skill / "SKILL.md").read_text()
    body = re.sub(r"---\n.*\n.*\n---\n\n", "", skill_text)

    for i, arg in enumerate(args):
        body = body.replace(f"$ARGUMENTS[{i}]", arg).replace(f"${i}", arg)

    return body.replace("$ARGUMENTS", " ".join(args))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    cli_args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    system_prompt = ""
    if SKILLS_PATH.exists():
        system_prompt += "You have access to the following skills:\n"
        for skill in SKILLS_PATH.iterdir():
            skill_text = (skill / "SKILL.md").read_text()
            system_prompt += "\n- " + ": ".join(
                re.search(
                    r"---\nname: (.*)\ndescription: (.*)\n---", skill_text
                ).groups()
            )
        system_prompt += "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."

    prompt: str = cli_args.p
    args = prompt.split(" ")

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    is_skill = lambda x: x.startswith("/")
    if is_skill(prompt):
        skills = list(itertools.takewhile(is_skill, args))
        args = list(itertools.dropwhile(is_skill, args))

        for skill in skills:
            messages.append({"role": "user", "content": apply_skill(skill[1:], args)})
    else:
        messages.append({"role": "user", "content": prompt})

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
                result = Path(args["file_path"]).read_text()
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            elif call.function.name == "Write":
                args = json.loads(call.function.arguments)
                Path(args["file_path"]).write_text(args["content"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Successfully wrote to file",
                    }
                )
            elif call.function.name == "Bash":
                args = json.loads(call.function.arguments)
                result = subprocess.run(
                    args["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                output = result.stdout + result.stderr
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": (
                            output if output else "Command completed with no output"
                        ),
                    }
                )
            elif call.function.name == "Skill":
                args = json.loads(call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": apply_skill(
                            args["name"],
                            args["args"].split(" ") if "args" in args else [],
                        ),
                    }
                )


if __name__ == "__main__":
    main()
