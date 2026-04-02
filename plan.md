i want to create  a system kaggleclaw this  is saas that run notebook env but also use ngrok for free url so we interract with model.  this use oss-120b and harmony for prompt adjustment.

---

workflow:
host model ->> kaggle notebook
lunch app ->> public url using ngrok
model access ->> web internt, python tool, jupyter notebook (i already create most of stuff like .agent/browser* & python and apply patch)

from start we give mannuly collect competition information in (competition.md, metrix.py) this two file model input and give model acess of internt and authority to use tools to create ,  test and solve this competition as you own 

for front end show extract what model does in conversation formate tool excuate , there output and model ganration and thinking show in profectinal way 
----

give model this:

/kaggle/working/run ->> this dir is test  experiments for model 
/kaggle/input/{} ->> competition dataset i add manuly in competition.md
/kaggle/working/. this where our all file have 

---

make a proper plan andcreate a unlimated Kaggleclaw that can solve any kaggle given competitionn as own!

-----------------------

Now i introduct models/host.py that host model in notebook optimazed way like  preloaded wights and then serve using vllm is i gues most fast and optimazed way 
help me to build correct host model adding corner button "Host model or  Model Hosted" this help to see hosted model 

correctly align data and api call though model using harmony and vllm , chat, browser , tool  check  every think and add correctly  


---

Error found 
1. Agent loop error: Failed to create harmony client: cannot import name 'Client' from 'openai_harmony' (/usr/local/lib/python3.12/dist-packages/openai_harmony/__init__.py)
2. Error starting server: 'Server' object has no attribute '_initialize_kernels'
Task exception was never retrieved
future: <Task finished name='Task-18' coro=<AgentRunner.send_user_message() done, defined at /kaggle/working/KaggleClaw/agent/run.py:148> exception=RuntimeError("Failed to create harmony client: cannot import name 'Client' from 'openai_harmony' (/usr/local/lib/python3.12/dist-packages/openai_harmony/__init__.py)")>

3. when model hosting the corner button we added this add host model , hosting.. , model hosted (this dynamic after model hosted then show model hosted and stop hosting button)
--
check hormony doc and  correctly implemnt harmony and prompt addintion use internet to find  all this

----------

Problem face while using:

1. convesion show error in frontend 
`User: hiHello! 👋 How can I help you today?User: who are youI’m ChatGPT, an AI language text, answer here to help you as best as I can! How can I assistSure thing! I’d love to get started— forward to the details!`

this happed when i type to conversion this not show correct converstion formate also not scollable 
- add scolable model & user conversation prompts
- when model ganration 
    - thinking
    - tool
    - main output

this also show is mendatory

2. when click rest whole even are reset all data collected that event even stop all proces immgetly reload model internally this rest mens whole reset not past / preview such thinks 

3. Error stopping vLLM: Command '['/usr/bin/python3', '-m', 'vllm.entrypoints.openai.api_server', '--seed', '42', '--model', '/kaggle/input/models/danielhanchen/gpt-oss-20b/transformers/default/1', '--served-model-name', 'oss-120b', '--tensor-parallel-size', '1', '--max-num-seqs', '256', '--gpu-memory-utilization', '0.92', '--host', '0.0.0.0', '--port', '8080', '--dtype', 'auto', '--kv-cache-dtype', 'auto', '--max-model-len', '65556', '--stream-interval', '1', '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching']' timed out after 10 seconds

correctly understad problem 

correct the fronntend and backend issue show clearly what goning on every model ganrated text showing is mandatory 
----

Okay so setup is almost complete but those is adding is mandatory help me to add this setup

- respone stop button (stop button located to send button in bottom)
- correct the converation formate in frontend 
    - so the converstion get longer it not show all text in screen , scolling not work, .md text not look like table and other thinks not show correctly 
    - we can visite harmony doc https://developers.openai.com/cookbook/articles/openai-harmony look tokens ganration so you correctly extract thinking, tool call and final ganaration 
    ```
    <|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
    <|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
    ```
    ```
    Special Tokens

The model uses a set of special tokens to identify the structure of your input. If you are using tiktoken these tokens are encoded in the o200k_harmony encoding. All special tokens follow the format <|type|>.
Special token	Purpose	Token ID
<|start|>	Indicates the beginning of a message. Followed by the “header” information of a message starting with the role	200006
<|end|>	Indicates the end of a message	200007
<|message|>	Indicates the transition from the message “header” to the actual content	200008
<|channel|>	Indicates the transition to the channel information of the header	200005
<|constrain|>	Indicates the transition to the data type definition in a tool call	200003
<|return|>	Indicates the model is done with sampling the response message. A valid “stop token” indicating that you should stop inference.	200002
<|call|>	Indicates the model wants to call a tool. A valid “stop token” indicating that you should stop inference.	200012
```


- adding a right side bar in that add a current project runing and small file setup show dir so we can interract how model 
look and create file 

- looking code for frontend what bug look wrong corect this with correct logic 

--------

Now this:

1. what current frontend show chat does is warped in conatiner (user or assistant ) and show so this cousing scolling error and text got text positition fixed and hideenflow hide - use normal chatgpt or chat interface so thos not create error 
2. when model typing any code this process whill stop i gues tool calling trigger which good but  stop is bug correct that use  tool call method like 
```
                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
    
                if last_message.channel == 'final':
                ...
    
                if last_message.recipient == 'python':
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last_message)
```

like this help to  get correct tool calling

---

when we start agent this process also show what model enter prompt  and what's goinig on
model  ganration same as chat

---
Error face 

Task exception was never retrieved
future: <Task finished name='Task-63' coro=<AgentRunner.run() done, defined at /kaggle/working/KaggleClaw/agent/run.py:77> exception=APIError('Unexpected token 12606 while expecting start token 200006')>
Traceback (most recent call last):
  File "/kaggle/working/KaggleClaw/agent/run.py", line 104, in run
    async for msg in stream_completion(
  File "/kaggle/working/KaggleClaw/agent/harmony.py", line 117, in stream_completion
    async for chunk in stream:
  File "/usr/local/lib/python3.12/dist-packages/openai/_streaming.py", line 153, in __aiter__
    async for item in self._iterator:
  File "/usr/local/lib/python3.12/dist-packages/openai/_streaming.py", line 200, in __stream__
    raise APIError(
openai.APIError: Unexpected token 12606 while expecting start token 200006