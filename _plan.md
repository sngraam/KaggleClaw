Act as Ml pipline expert and agentic work flow creator

# what i'm bulding

---
I'm building kaggleclaw a system that work in kaggle notebook and solve ml project using oss-120b model with1 h100 gpu self improving as there own making file and desicians expeertmenst and test cases as own building understading and decodee scoring metrix

---

for this i'm create this codebase to matching the way i build but in this theree are some gaps i can't fill with you help me to build this autonoums system 

---

My codebase is still devloping  stage and heelp me to improve logic correct handling task and solve the problem while constaly create new eexperimeents experiment 

---

# backend
i'm created 5-6 tools for model usages agent/tools/* 
- some system prompt agent/command/* 
- main model AgentRunner class in run.py

Task 0: understad the backend , model hosting system and how i handel tokens ganrated , time complexity and how to improve it.

- create 512 tokens buffer for ganrated token so system not get lacked use advance methoad to correctly show model output in order to not crash and manage all function correctly 

- Understad workflow and modified ganration process:

```
## workflow

- model frist task is get the data competition.md & metric.py code and create a EDA report while using tools reserch web and dataset create a full EDA that help to understand the data and problem statement in last frist task model save EDA.md file in current dir/run/eda.md like formate 
    - create syatem prompt for eda of kaggle any competition 
    - correctly accese tool and all infromation

- After Eda model move to sencond phase is model get (system + EDA data) and we tell that create a full wining pipline code for this competition test cases and see the output improve over the output!

- and after the train pipline get output we simply tell model to improve val scoring also accoding to CV LB management gap!

```

Task 1: Manage Aync fuction and process that code not get crash, model not get stuck and trap in loop 
also see the process and correct with understad nature of python code and model ganration when model done or not!

Task 2: Handel api endpoint and server issue spend time on server/* and fix it make scalable and fast correctly handel conditition and model ganration

what we give: we give model tool and 65k - input token buget so model find there answer as he own i accept if front end get dealy but model don't need correct prompt and information and correct promting and full process we don't stop model if is not done to exproing !

# frontend
i'm created simple ui for agent interaction agent/app.py

Task: in frontend you see right side bar space of bottom in this add a terminial logger that show what goning on and if fail any moment termial get show why! also add copy button so i simply copy and paste here !!


# Model Host
i'm currently use vllm to host model like server that i call api and model repones !

----

Your main focus is backend model serve to input --> output -> frontend show also tool output and thinks not model get stop unexpectly !!

# create command prompt 

create some command/* prompt that use in phase like EDA ->> pipline ganration ->> model training ->> model evaluation like this help model to correct instruction of that data 

Build correct and valueable, scalable and fast backend for kaggle claw system !!