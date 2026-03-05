import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIResponsesClient
from dotenv import load_dotenv
from rich import print
from rich.logging import RichHandler

# Setup logging
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure OpenAI client based on environment
load_dotenv(override=True)

client = OpenAIResponsesClient(
    api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    )

@tool
def run_script(script_path: str) -> str:
    """Execute a bash script at the given path and return its output."""
    logger.info("run script tool")
    result = subprocess.run(
        ["bash", script_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout
    if result.returncode != 0:
        output += f"\nSTDERR: {result.stderr}"
    return output or "(no output)"

coder_agent = Agent(
    client=client,
    instructions=(
        "You are a bash scripting expert on macOS. "
        "Your only job is to write a single, plain bash script that solves the given task. "
        "Rules you must follow:\n"
        "- Output ONLY the raw bash script, starting with #!/bin/bash\n"
        "- Do NOT include markdown code fences (``` or ```bash)\n"
        "- Do NOT include any explanation, comments, or prose before or after the script\n"
        "- Do NOT use Python, Ruby, or any other language — bash only\n"
        "- The script must run without any user input\n"
        "- Write for macOS only — do NOT include Windows or Linux alternatives\n"
        "Example of correct output:\n"
        "#!/bin/bash\nfind ~ -name '*.txt'\n"
    ),
    tools=[],
)

@tool
async def coder(query: str) -> str:
    """Write a bash script to solve the task and return its file path."""
    logger.info ("write script to file tool")
    response = await coder_agent.run(query)

    # Strip markdown code fences if present
    code = re.sub(r"```[a-z]*\n", "", response.text)
    code = re.sub(r"```", "", code).strip()
    # Ensure shebang line
    if not code.startswith("#!"):
        code = "#!/bin/bash\n" + code
    script_path = os.path.join(tempfile.gettempdir(), f"script_{uuid.uuid4().hex}.sh")
    with open(script_path, "w") as f:
        f.write(code)
        logger.info("---")
        logger.info(code)
    os.chmod(script_path, 0o755)

    return script_path

# SCRIPT LOCAL AGENT

developer_agent = Agent(
    name="developer_agent",
    client=client,
    instructions=(
        "You are an execution agent on macOS. "
        "Your job is to generate a bash script for the given task and execute it. "
        "Always call the coder tool first to generate the script, then call run_script with the returned path. "
        "Return ONLY the raw output from run_script — no explanation, no commentary."
    ),
    tools=[coder, run_script],
)

@tool
async def developer(query: str) -> str:
    """Generate and execute a bash script on the local machine to solve the task. Returns the raw script output."""
    response = await developer_agent.run(query)
    return response.text

# ROOT AGENT

root_agent = Agent(
    name="root",
    client=client,
    instructions=(
        "You are a helpful assistant. "
        "If the user's request involves anything on his local machine, pass the question to the developer tool"
        "DO NOT WRITE ANY CODE YOURSELF"
        "For all other requests, answer directly from your own knowledge."
    ),
    tools=[developer],
)

# MAIN LOOP

async def main(user_query: str):
    response = await root_agent.run(user_query)
    logger.info("response start")
    print(response.text)
    logger.info("response end")

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[root_agent], auto_open=True)
    else:
        asyncio.run(main(sys.argv[1]))
