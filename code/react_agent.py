import openai
from typing import List, Dict, Callable

openai.api_key = ""
# openai.base_url

def format_tools_for_prompt(available_tools: List[Dict]) -> str:
    """Formats the tool definitions from ToolHop into a string for the prompt."""
    prompt_str = "You have access to following tools:\n"
    for tool_spec in available_tools:
        prompt_str += f"- {tool_spec['name']}: {tool_spec['description']}"
    return prompt_str

def rum_my_react_agent(query: str, available_tools: List[Dict]) -> str:
    """
        Executes the ReAct logic for a given query and set of tools.

        Args:
            query: The user's question.
            available_tools: A list of tool specifications from the ToolHop dataset.

        Returns:
            The final answer string.
    """
    prompt_history = f"Question: {query}\n"
    prompt_history += format_tools_for_prompt(available_tools=available_tools)
    prompt_history += "Begin!"

    max_turns = 7
    for _ in range(max_turns):
        prompt = f"""
            {prompt_history}
            Based on history what is your next thought and action?
            Strictly follow the format:
            Though: [Your reasoning here]
            Action: [tool_name(argument)] or Final Answer: [Your Final Answer]
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        response_text = response.choices[0].message.content.strip()

        if "Final Answer:" in response_text:
            final_answer = response_text.split("Final Answer:")[-1].strip()
            return final_answer

        try:
            thought = response_text.split("Thought:")[-1].split("Action:")[0].strip()
            action_str = response_text.split("Action:")[-1].strip()

            tool_name, argument = action_str.split("(", 1)
            argument = argument.rstrip(')')
        except Exception as e:
            observation = f"Error parsing action: {e}. Please try again."
            prompt_history += f"\nThought: {thought}\nAction: {action_str}\nObservation: {observation}"
            continue
