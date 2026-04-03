

from _agent.tools import Tool, Sandbox, FileTool
from harmo import HarmoTemplate

class Claw:
    def __init__(self, env):
        self.env = env
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = HarmoTemplate()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.python = Tool(
            local_jupyter_timeout=120.0,
            tool_prompt=None,
            sandbox=Sandbox(
                timeout=120.0
            )
        )

        self.fileread = FileTool()
        self.patch = ApplyPatchTool()

    def grab(self):
        return self.env.grab()

    def release(self):
        return self.env.release()
    
    async def stream_completion(
        self, 
        messages: list,
        model: str,
        event_queue: asyncio.Queue | None = None,
    ) -> AsyncIterator:
        
        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        
        logprobs_buffer = []
    
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))
    
        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
    
            local_tool = self.python
    
            encoding = self.encoding
            messages = self.template.apply_chat_template(
                system_prompt, 
                problem, 
                local_tool.tool_config
            )
    
            conversation = Conversation.from_messages(messages)
    
            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break
    
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
    
                if max_tokens < self.cfg.buffer_tokens:
                    break
    
                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, 
                    temperature=self.cfg.temperature, 
                    logprobs=self.cfg.top_logprobs, 
                    max_tokens=max_tokens, 
                    prompt=prompt_ids, 
                    seed=attempt_seed, 
                    stream=True, 
                    extra_body={
                        'min_p': self.cfg.min_p, 
                        'stop_token_ids': self.stop_token_ids, 
                        'return_token_ids': True
                    }
                )