from typing import List, Type, Union
from groq import Groq
from openai import OpenAI
from instructor.client import Instructor
from pydantic import BaseModel
from ratelimit import limits, sleep_and_retry
import logging

#TODO: Change this to a routing function for other providers
#OpenAI format will do for 99% providers

# Initializing logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# "Normal" text response
@sleep_and_retry
@limits(calls = 1, period = 0.5)
def get_response(messages: List[dict], 
                 client: Union[Groq, OpenAI], 
                 client_args: dict, **kwargs):
    response = client.chat.completions.create(
        messages = messages,
        **client_args
    )
    response_dict = dict(response.choices[0].message)
    response_dict.pop('function_call', None)
    response_dict.pop('tool_calls', None)
    return response_dict

# Structured Response
# Avoids exceeding call limit
@sleep_and_retry
@limits(calls = 1, period = 0.5)
def get_structured_response(messages: List[dict],
                            response_model: Type[BaseModel] = None,
                            return_fields: List[str]| str | None = ["response"],
                            single_item_list_return_dict: bool = False,
                            client: Instructor = None,
                            client_args: dict = dict(),
                            verbose: bool = False,
                            **kwargs):
    response: BaseModel = client.chat.completions.create(
        response_model = response_model,
        messages = messages,
        **client_args
    )
    if verbose:
        logger.info(str(response))
    stripped_response = response_fields(response, return_fields, single_item_list_return_dict)
    return stripped_response

def response_fields(response: BaseModel, return_fields: List[str]| str | None, single_item_list_return_dict: bool):
    if return_fields is None or len(return_fields) == 0:
        return response

    if isinstance(return_fields, str):
        return getattr(response, return_fields)

    if len(return_fields) == 1 and not single_item_list_return_dict:
        return getattr(response, return_fields[0])

    return {field: getattr(response, field) for field in return_fields}