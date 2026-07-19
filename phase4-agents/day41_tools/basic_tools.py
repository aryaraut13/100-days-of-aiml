import os
import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Input should be a valid Python math expression like '2 + 2' or 'math.sqrt(16)'.
    """
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words, characters, and sentences in a text.
    """
    words      = len(text.split())
    characters = len(text)
    sentences  = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words} | Characters: {characters} | Sentences: {sentences}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    Supported: km/miles, kg/pounds, celsius/fahrenheit
    """
    conversions = {
        ("km", "miles"):       lambda x: x * 0.621371,
        ("miles", "km"):       lambda x: x * 1.60934,
        ("kg", "pounds"):      lambda x: x * 2.20462,
        ("pounds", "kg"):      lambda x: x / 2.20462,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."


if __name__ == "__main__":
    print("[TOOL DEFINITIONS]\n")
    print(f"Tool 1: {calculator.name}")
    print(f"  Description: {calculator.description}\n")

    print(f"Tool 2: {word_counter.name}")
    print(f"  Description: {word_counter.description}\n")

    print(f"Tool 3: {unit_converter.name}")
    print(f"  Description: {unit_converter.description}\n")

    print("[TOOL OUTPUTS]\n")
    print(calculator.invoke("math.sqrt(144) + 2**3"))
    print(word_counter.invoke("AI agents are autonomous systems that use tools to accomplish tasks."))
    print(unit_converter.invoke({"value": 100, "from_unit": "km", "to_unit": "miles"}))