from pydantic import BaseModel
from groq import Groq
from typing import List, Any

# Standard models
class BooleanModel(BaseModel):
    thoughts: str
    response: bool # If the user is asking or not

class ListStrModel(BaseModel):
    thoughts: str
    response: List[str] # List of new queries


# Custom models
class HyDE(BaseModel):
    thoughts: str
    generate: bool
    response: str