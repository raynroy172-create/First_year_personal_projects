from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

client = OpenAI()

system_prompt = '''
You are an autonomous CLI Agent operating directly within a user's terminal environment. Your purpose is to assist the user by planning tasks, executing system commands, and providing clear feedback.

You operate using a strict Chain of Thought process. You must process one step at a time, allowing the system to execute your commands and append the results to the message history before you continue.

AVAILABLE TOOLS:
- run_command: Executes a shell command in the system. 
  Argument format: string (the exact command to run, e.g., "ls -la" or "cat file.txt").

STRICT OUTPUT FORMAT:
You must respond with EXACTLY ONE valid JSON object per turn. Do not wrap the JSON in markdown code blocks (no ```json ... ```). Do not include any conversational filler, greetings, or text outside of the JSON object. Failure to output raw, valid JSON will crash the system.

Your JSON output must strictly adhere to the following schema:
{
  "step": "<must be one of: 'start', 'plan', 'output'>",
  "content": "<your thought process, reasoning, plan, or the message you want to display to the user>",
  "tools": "<'run_command' if you need to execute a command, otherwise null>",
  "arguments": "<the string command to execute if tools is 'run_command', otherwise null>"
}

EXECUTION LOOP (Chain of Thought):
1. PLAN: When given a new task, your first response must be a "plan" step (tools = null) where your content details the exact steps you will take.
2. EXECUTE: Next, issue an "output" step where tools = "run_command" and arguments contains the command. The system will pause your execution, run the command, and provide the terminal output in the next message.
3. ITERATE: Continue issuing "run_command" steps one at a time until you have achieved the user's goal. Analyze the output of each command in your subsequent "content" fields.
4. FINISH: Once the task is fully complete, provide a final "output" step (tools = null, arguments = null) where the content summarizes the completed work for the user.

Remember: Output ONLY the raw JSON object. One JSON object per turn.
'''

def run_command(cmd:str):
    result = os.system(cmd)
    return(result)

get_tools = {'run_command':run_command}
message_history = [{'role':'system', 'content':system_prompt}]

while True:
    prompt = input('Enter a prompt ➡️ ')
    message_history.append({'role':'user','content':prompt})

    while True:
        response = client.chat.completions.create(
            model='gemini-2.5-flash',
            messages= message_history,
            response_format = {'type':'json_object'}
        )

        model_output = response.choices[0].message.content
        message_history.append({'role':'assistant', 'content':model_output})

        parsed_output = json.loads(model_output)

        if parsed_output.get('step') == 'plan':
            print(f'🧠{parsed_output.get('content')}')
            message_history.append({'role':'user', 'content':'Plan acknowledged. Execute the first step.'})

        
        elif parsed_output.get('step') == 'output':
            print(f'✅👉🏻{parsed_output.get('content')}')

            tool_name = parsed_output.get('tools')
            tool_args = parsed_output.get('arguments',{})

            if tool_name == 'run_command':
                cmd = tool_args
                out = get_tools['run_command'](cmd)
                message_history.append({'role':'user', 'content':f'Command executed with exit code status {out}'})
            
            else:
                break
