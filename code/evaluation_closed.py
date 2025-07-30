"""
 Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """


import argparse
import json
import os
from hashlib import sha256
import datetime
from tqdm import tqdm
from func_timeout import func_set_timeout
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import requests


def req_closed(messages, model='gpt-4o-2024-08-06', temperature=0., base_url=None, api_key=None, max_tokens=256, **kwargs):
    t = 0
    while t < 3:
        try:
            logid = sha256(messages[0]['content'].encode()).hexdigest()
            headers = {
                'Content-Type': 'application/json',
                'X-TT-LOGID': logid,
                'Authorization': f'Bearer {api_key}'
            }
            data = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            response = requests.post(
                f'{base_url}', headers=headers, json=data, timeout=30)

            return response.json()
        except Exception as e:
            t += 1
            print("error req_closed", response, e, flush=True)
            sleep(5)
    return None


def test_closed(messages, args, tools=None):
    try:
        response = req_closed(messages=messages, model=args.model, temperature=0., tools=tools,
                              base_url=args.base_url, api_key=args.api_key, max_tokens=256)
        return response['choices'][0]['message']
    except Exception as e:
        print("error test closed" , response, e, flush=True)

    return None


def get_feedback(tool_calls, functions):
    @func_set_timeout(10)
    def excute_func(functions, func_name, func_args):
        for func in functions:
            if func.split('def')[1].split('(')[0].strip() == func_name:
                exec(func)
        return eval(func_name)(**func_args)

    res = []
    for tool_call in tool_calls:
        try:
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])

            feedback = excute_func(functions, tool_name, tool_args)

            res.append({"role": "tool", "content": json.dumps(
                feedback, ensure_ascii=False), "tool_call_id": tool_call["id"]})
        except Exception as e:
            res.append(
                {"role": "tool", "content": f"an error occured when call {tool_call['function']['name']}: {str(e)}", "tool_call_id": tool_call["id"]})

    return res


def sample_process_direct(sample, args):
    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    response = test_closed(messages=messages, args=args)
    if response:
        messages.append(response.copy())
        sample['messages'] = messages.copy()
        return sample
    return None


def sample_process_mandatory(sample, args):
    turn = 0

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]
    while turn < args.max_turns:
        response = test_closed(messages=messages, tools=tools, args=args)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                feedback = get_feedback(
                    response['tool_calls'], sample['functions'])
                messages += feedback.copy()
            if not response.get('tool_calls', None) or turn == args.max_turns - 1:
                sample['messages'] = messages.copy()
                sample['turn'] = turn
                return sample

        turn += 1
    return None


def sample_process_free(sample, args):
    turn = 0

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]
    while turn < args.max_turns:
        response = test_closed(messages=messages, tools=tools, args=args)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                feedback = get_feedback(
                    response['tool_calls'], sample['functions'])
                messages += feedback.copy()
            if not response.get('tool_calls', None) or turn == args.max_turns - 1:
                sample['messages'] = messages.copy()
                sample['turn'] = turn
                return sample

        turn += 1
    return None


def sample_process(sample, args):
    if args.scenario == "Direct":
        sample = sample_process_direct(sample, args)
    elif args.scenario == "Mandatory":
        sample = sample_process_mandatory(sample, args)
    elif args.scenario == "Free":
        sample = sample_process_free(sample, args)
    else:
        raise NotImplementedError

    if sample:
        if sample['messages'][-1]['role'] == 'assistant':
            if sample['messages'][-1]['content']:
                if sample['answer'].rstrip('.0').lower() in sample['messages'][-1]['content'].rstrip('.0').replace(',', '').lower():
                    sample['answer_correct'] = 1
                elif sample['messages'][-2]['role'] == 'tool' and sample['answer'].rstrip('.0').lower() in sample['messages'][-2]['content'].rstrip('.0').replace(',', '').lower():
                    sample['answer_correct'] = 1
                else:
                    sample['answer_correct'] = 0

    return sample


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="Direct",
                        choices=["Direct", "Mandatory", "Free"])
    parser.add_argument("--series", type=str, default="gpt",
                        choices=["gemini15", "claude35", "gpt"])
    parser.add_argument("--model", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--input_file", type=str,
                        default="data/ToolHop.json")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument(
        '--max_turns', type=int, default=9)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    with open(args.input_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids = [json.loads(line)["id"] for line in f.readlines()]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(args.output_file, 'a') as f:
        futures = [executor.submit(sample_process, sample, args)
                   for sample in data if sample["id"] not in ids]
        for future in tqdm(as_completed(futures), total=len(futures)):
            response = future.result()
            if response:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
                f.flush()

    with open(args.output_file, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f.readlines()]
        total_sum = 0
        for sample in data:
            try:
                total_sum += sample["answer_correct"]
            except Exception as e:
                e;

        result = round((100 * total_sum) / 995, 2)
        print(
            f"Scenario: {args.scenario}\nSeries: {args.series}\nModel: {args.model}\nValid Items: {len(data)}\nResult: {result}\n")


if __name__ == "__main__":
    main()
