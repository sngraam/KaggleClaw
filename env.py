import os
from pyngrok import ngrok
import threading
import time

CONF_TOKEN = "2oWmSCgxzNQCpZllXwZR5xoCebZ_3bsJqtwJRZ6LzPkhasfTx"
ngrok.set_auth_token(CONF_TOKEN)


MODEL_PATH = "/kaggle/input/oss-120b"
# GPU 1 h100

