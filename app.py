import datetime

import pytz
import requests
import yaml
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    TransformersModel,
    LiteLLMModel,
    load_tool,
    tool,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from Gradio_UI import GradioUI
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool


# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(
    arg1: str, arg2: int
) -> str:  # it's import to specify the return type
    # Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
visit_web_page = VisitWebpageTool()
duck_duck_go_search = DuckDuckGoSearchTool()

# If the agent does not answer, the model is overloaded,
# please use another model or the following Hugging Face Endpoint
# that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'

# model = HfApiModel(
#     max_tokens=2096,
#     temperature=0.5,
#     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # it is possible that this
#     # model may be overloaded
#     custom_role_conversions=None,
# )

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

# model = TransformersModel(model_id=model_name, max_new_tokens=4096, device_map="auto")

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:32b",
    # api_base="http://localhost:3000",
    # api_key="sk-f92261f18fe34d088c5e045178792e21",
    num_ctx=8192
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype="auto", device_map="auto"
# )


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    # add your tools here (don't remove final answer)
    tools=[
        visit_web_page,
        duck_duck_go_search,
        get_current_time_in_timezone,
        image_generation_tool,
        final_answer,
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
)


GradioUI(agent).launch()
