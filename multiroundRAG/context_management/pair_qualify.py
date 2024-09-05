from pydantic import BaseModel
from groq import Groq
from typing import Annotated, List, Any, Callable
from annotated_types import Len
from .agent_prompts import qualify_sysprompt, qualify_prompt

# Post processing for qualification of existing pairs
# If max length per split = 0 then will process all pairs at once
# the additional arguments of the response_func will have been partialed into it when initialization starts
def qualify_existing_pairs(pairs: List[Any], chat_history: List[dict], response_func: Callable, max_length_per_split: int = 0):
    # No processing if empty:
    if not pairs:
      return [], []

    # Assumes "pairs" list is not empty
    splitted = split_list(pairs, max_length_per_split)
    split_results = [qualify_pairs(pairs_list, chat_history, response_func) for pairs_list in splitted]
    qualified = [pair for sublist in split_results for pair in sublist]
    return [pair for pair, qual in zip(pairs, qualified) if qual], [pair for pair, qual in zip(pairs, qualified) if not qual]


def split_list(pairs: List[Any], max_length_per_split) -> List[List[dict]]:
    # Check No splitting
    if not max_length_per_split:
        return pairs
    return [pairs[i:i+max_length_per_split] for i in range(0, len(pairs), max_length_per_split)]

def qualify_pairs(pairs: List[Any],
                  chat_history: List[dict],
                  response_func: Callable) -> List[bool]:
    num_pairs = len(pairs)
    msg = [qualify_sysprompt(num_pairs)] + chat_history + [qualify_prompt(pairs)]
    response_model = batch_qualify(num_pairs)
    response = response_func(
        response_model = response_model,
        messages = msg
    )
    return response


def batch_qualify(batch_length: int):
    class BatchQualify(BaseModel):
        thoughts: str
        response: Annotated[List[bool], Len(min_length=batch_length, max_length=batch_length)]

    return BatchQualify

# from pydantic import BaseModel, validator, ValidationError
# from groq import Groq
# from typing import List

# def qualify_all_pairs(pairs: List[dict], chat_history: List[dict], client: Groq) -> List[bool]:
#     return [qualify_pair(pair, chat_history, client) for pair in pairs]

# # Fallback, if system cannot output fixed length list
# class Qualify(BaseModel):
#     thoughts: str
#     qualify: bool

# def qualify_pair(pair: dict, chat_history: List[dict], client: Groq) -> bool:
#     client_input = [qualify_sysprompt] + chat_history + [qualify_prompt(pair)]
#     response: Qualify = client.chat.completions.create(
#         response_model = Qualify,
#         messages = client_input,
#         **groq_args
#     )

#     return response.qualify

# def qualify_prompt(pair: dict) -> str:
#     msg = {
#         "role": "User",
#         "msg": (f"Consider the following query-retrieved pair: {pair}."
#                 "is it relevant to the user's last message?")
#     }