import dspy
import os
import tqdm

lm = dspy.LM(
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506", api_key="", api_base="https://api.deepinfra.com/v1/openai"
)

dspy.configure(lm=lm)

agent = dspy.ReAct(tools=[], signature="question->answer")
