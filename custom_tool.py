
from turtle import st

from crewai_tools import tool

@tool("find_palindromes")

def find_palindromes(sequence: str)  -> list[str]
    return 