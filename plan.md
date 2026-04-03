I'm building kaggleclaw a system that work in kaggle notebook and solve ml project using oss-120b model with1 h100 gpu self improving as there own making file and desicians expeertmenst and test cases as own building understading and decodee scoring metrix

---

for this i'm create this codebase to matching the way i build but in this theree are some gaps i can't fill with you help me to build this autonoums system 

# backend
---
agent/tools/* 

- frist we most avalable tool and there qulity the current i corrctly made is statefull py excuation tool that help model to small test cases i also want to crete this tool help me to build this 

FileTool - file read, write, edit using patch,
WebSearchTool - serch web collect information and data model want (free no api needed)

PythonTool - already  created just see any improvemnets needed

Patch - this correctly modified edit file using patches

PlanFollow - follow created plan for solving current competition
for now we focus this 5 tool becuse this are main tool to run intract model 

Note:
for tool correctly trigger model ganarated tool call this 
```
                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
    
                if last_message.channel == 'final':
                ...
    
                if last_message.recipient == 'python':
                ...
                if last_message.recipient == 'file':
                ...
                if last_message.recipient == 'apply_patch':
                ...
```
>```assistantcommentary to=functions.apply_patch```
& for pytool ONLY
```assistantanalysis to=python```
why becuse functions prompt in devloper call functions.{name} and pytool is system that way is direct call analysis to=python channel

---

agent/command/*

- here are all model prompt (fancy word skill prompt) add prompt devloper or system or user that model follow 

---
expected workflow
Plan: 
->> get competition.md & metric.py (competition information)
->> create eda and save --> (eda + past info) -> create plan
->> follow plan and build report that experiment current working 
(eg., run/exp1/eda.md, comp.md, metric.py, report.md, train.py ..)
---

# frontend
1.
correctly adjust ganrated tokens to decode and the show frontend
("/n", latex or tables  , ##, ###, # like this correctly show)

2. extract  information token wise and represet correct formate 

```python
Special token	Purpose	Token ID
<|start|>	Indicates the beginning of a message. Followed by the “header” information of a message starting with the role	200006
<|end|>	Indicates the end of a message	200007
<|message|>	Indicates the transition from the message “header” to the actual content	200008
<|channel|>	Indicates the transition to the channel information of the header	200005
<|constrain|>	Indicates the transition to the data type definition in a tool call	200003
<|return|>	Indicates the model is done with sampling the response message. A valid “stop token” indicating that you should stop inference.	200002
<|call|>	Indicates the model wants to call a tool. A valid “stop token” indicating that you should stop inference.	200012
```

Note: Model not use 200012 token every time call so also handel this condition

correctly manage advance steming system that my browser not steep of high speed token and show all token in order to not take damage 

3. for evary model ganration token so some how the frontend in approcheble formate so we correctly debug all process steep-by-step 
---

# Error handeling 

becuse of system is much more compelex print the errror happed in front end or log so we know where and what is error !!

Doc: https://developers.openai.com/cookbook/articles/openai-harmony


act as an advance ml agent buldinger engnner aand vuild this advance system with you undersading 