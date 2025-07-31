import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from pydantic import create_model, BaseModel, Field

JSON_TYPE_TO_PYTHON_TYPE = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list
}

def create_dynamic_langchain_tools(functions, tool_definitions):
    """
    Creates a list of LangChain Tool objects from the raw function code and definitions.
    """
    tools = []

    tool_codes = {}
    for func_str in functions:
        try:
            func_name = func_str.split('def ')[1].split('(')[0].strip()
            tool_codes[func_name] = func_str
        except IndexError:
            continue

    for _, tool_def in tool_definitions.items():
        tool_name = tool_def['name']
        if tool_name in tool_codes:

            fields = {}
            tool_params = tool_def.get('parameters', {})
            required_params = tool_params.get('required', [])

            if 'properties' in tool_params:
                for param_name, param_props in tool_def['parameters']['properties'].items():
                    param_type = JSON_TYPE_TO_PYTHON_TYPE.get(param_props.get('type', 'string'), str)
                    param_description = param_props.get('description', '')
                    if param_name in required_params:
                        fields[param_name] = (param_type, Field(description=param_description))
                    else:
                        fields[param_name] = (Optional[param_type], Field(default=None, description=param_description))

            args_schema = create_model(f'{tool_name}Args', **fields, __base__=BaseModel)

            def create_tool_func(name, code):
                def _tool_func(**kwargs):
                    try:
                        local_scope = {}
                        exec(code, globals(), local_scope)
                        result = local_scope[name](**kwargs)
                        return result
                    except Exception as e:
                        return f"Error executing tool {name}: {str(e)}"
                return _tool_func

            tools.append(
                StructuredTool(
                    name=tool_name,
                    description=tool_def['description'],
                    func=create_tool_func(tool_name, tool_codes[tool_name]),
                    args_schema=args_schema,
                    handle_tool_error=True
                )
            )
    return tools


def run_langchain_on_sample(sample, args):
    """
    Processes a single sample from the ToolHop dataset using a LangChain agent.
    """
    question = sample['question']

    try:
        llm = ChatOpenAI(
            model=args.model,
            temperature=0.0,
            api_key=args.api_key,
            base_url=args.base_url,
            max_completion_tokens=1024
        )
    except Exception as e:
        print(f"Failed to initialize ChatOpenAI: {e}")
        return None

    predicted_answer = ""
    try:
        if args.scenario == "Direct":
            prompt = f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {question}"
            response = llm.invoke(prompt)
            predicted_answer = response.content

        elif args.scenario in ["Mandatory", "Free"]:
            tools = create_dynamic_langchain_tools(sample['functions'], sample['tools'])
            if not tools:
                raise ValueError("No valid tools could be created for this sample.")


            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. You will be asked a question and given access to a set of tools. You must use the tools to find the answer. Provide the final answer in the requested format: <answer>your final answer here</answer>"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_openai_tools_agent(llm, tools, prompt)

            agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=args.max_turns, verbose=True)

            result = agent_executor.invoke({"input": question})
            predicted_answer = result['output']

    except Exception as e:
        print(f"Error processing sample {sample['id']} with LangChain: {e}")
        predicted_answer = ""

    sample['predicted_answer'] = predicted_answer
    sample['answer_correct'] = 0
    if predicted_answer:
        ground_truth = str(sample['answer']).rstrip('.0').lower()
        cleaned_prediction = predicted_answer.replace(',', '').lower()
        if '<answer>' in cleaned_prediction:
            cleaned_prediction = cleaned_prediction.split('<answer>')[-1].split('</answer>')[0].strip()

        if ground_truth in cleaned_prediction:
            sample['answer_correct'] = 1

    return sample


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ToolHop evaluation using LangChain.")
    parser.add_argument("--scenario", type=str, default="Mandatory",
                        choices=["Direct", "Mandatory", "Free"])
    parser.add_argument("--model", type=str, required=True, help="Model identifier passed to the API.")
    parser.add_argument("--base_url", type=str, required=False, help="API base URL for the OpenAI-compatible endpoint.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the endpoint.")
    parser.add_argument("--input_file", type=str,
                        default="data/ToolHop.json", help="Path to ToolHop.json")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output results.")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument('--max_turns', type=int, default=9, help="Max iterations for the LangChain agent.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    with open(args.input_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    ids_to_skip = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids_to_skip = {json.loads(line)["id"] for line in f}

    tasks = [sample for sample in data if sample["id"] not in ids_to_skip]

    if not tasks:
        print("All samples have already been processed.")
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(args.output_file, 'a', encoding='utf8') as f:
            futures = [executor.submit(run_langchain_on_sample, sample, args) for sample in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing with {args.max_workers} worker(s)"):
                response = future.result()
                if response:
                    f.write(json.dumps(response, ensure_ascii=False) + '\n')
                    f.flush()

    print("\nCalculating final results...")
    if not os.path.exists(args.output_file):
        print("No output file found. Cannot calculate results.")
        return

    with open(args.output_file, 'r', encoding='utf8') as f:
        results_data = [json.loads(line) for line in f]

    total_sum = sum(sample.get("answer_correct", 0) for sample in results_data)
    total_items = len(results_data)

    if total_items == 0:
        print("No valid results to calculate accuracy from.")
        return

    result_percentage = (100 * total_sum) / total_items

    print("\n" + "="*30)
    print(f"Framework: LangChain")
    print(f"Scenario: {args.scenario}")
    print(f"Model: {args.model}")
    print(f"Valid Items Processed: {total_items}")
    print(f"Correct Answers: {total_sum}")
    print(f"Final Accuracy: {result_percentage:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
