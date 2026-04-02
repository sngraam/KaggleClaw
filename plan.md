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