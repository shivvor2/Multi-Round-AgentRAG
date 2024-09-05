# Returns a dictionary with the fields "generate" and "response"
from typing import Callable
from .agent_prompts import hyde_sysprompt, hyde_prompt
from .response_models import HyDE 

def get_HyDE(query: str, response_func: Callable) -> dict:
    msg = [hyde_sysprompt] + [hyde_prompt(query)]
    response: HyDE = response_func(
        messages = msg,
        response_model = HyDE,
        return_fields = ["generate", "response"]
    )
    return response