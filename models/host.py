import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from openai import OpenAI

try:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
except ImportError:
    pass

@dataclass
class ServerConfig:
    model_path: str = "/kaggle/input/oss-120b"
    served_model_name: str = "oss-120b"
    workers: int = 4
    seed: int = 42
    batch_size: int = 256
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    context_tokens: int = 4096
    stream_interval: int = 1
    server_timeout: int = 600
    session_timeout: int = 120

class Server:

    def __init__(self, cfg, port: int = 8000):
    
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        try:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        except NameError:
            self.stop_token_ids = []

    
        self._preload_model_weights()
        
        self.server_process = self._start_server()
    
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )
    
        self._wait_for_server()
    
    def _preload_model_weights(self) -> None:
    
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        
        files_to_load = []
        total_size = 0
    
        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
    
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)
    
        def _read_file(path: str) -> None:
    
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')
    
    def _start_server(self) -> subprocess.Popen:
    
        cmd = [
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
            '--async-scheduling', 
            '--disable-log-stats', 
            '--enable-prefix-caching'
        ]
    
        self.log_file = open('vllm_server.log', 'w')
    
        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )
    
    def _wait_for_server(self):
    
        print('Waiting for vLLM server...')
        start_time = time.time()
    
        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
    
            if return_code is not None:
                self.log_file.flush()
    
                with open('vllm_server.log', 'r') as log_file:
                    logs = log_file.read()
    
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
    
            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
    
                return
    
            except Exception:
                time.sleep(1)
    
        raise RuntimeError('Server failed to start (timeout).\n')