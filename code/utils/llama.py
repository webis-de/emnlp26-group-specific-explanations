import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import logging
import time
from typing import List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding  # type: ignore

import re
import json
import deepspeed

from helper import log_time

logger = logging.getLogger(__name__)


class Llama:
    def __init__(
        self,
        model_id: str,
        enabled=True,
    ):
        self.enabled = enabled
        self.model_id = model_id
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if not enabled:
            return

    def __del__(self):
        if self.tokenizer is not None:
            del self.tokenizer
        if not self.enabled:
            return

        # logger.debug("Deleting Llama model")
        # from vllm.model_executor.parallel_utils.parallel_state import (  # type: ignore
        #     destroy_model_parallel,
        # )
        import gc
        import torch

        # destroy_model_parallel()
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            #torch.distributed.destroy_process_group()
            gc.collect()

    def generate(
        self,
        prompts: List[str],
        n_iterations=1,
        system_prompt: Union[str, None] = None,
        random_seed=42,
        return_prompts=False,
    ) -> List[List[str]] | Tuple[List[List[str]], List[str]]:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        base_chat_template = (
            [{"role": "system", "content": system_prompt}]
            if system_prompt is not None
            else []
        )
        prompts = [
             re.sub(
                r"Cutting Knowledge Date:.*?\nToday Date:.*?\n+",  # match the two lines
                "",
                self.tokenizer.apply_chat_template(
                    [*base_chat_template, {"role": "user", "content": x}],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                flags=re.S
            )
            for x in prompts
        ]
        
        print(prompts[0])
        print("random_seed", random_seed)
        time.sleep(10)
        answers = self._generate(
            prompts=prompts,
            n_iterations=n_iterations,
            random_seed=random_seed,
        )
        return answers if not return_prompts else (answers, prompts)
    
    def _generate(
        self,
        prompts: List[str],
        n_iterations=1,
        random_seed: Union[int, None] = None,
    ) -> List[List[str]]:
        if not self.enabled:
            answers = [
                ["The author Llama is disabled" for _ in range(n_iterations)]
                for _ in prompts
            ]
            return answers

        from vllm import SamplingParams
        from vllm import LLM

        if self.model is None:
            logger.debug(f"Loading Llama model (id {self.model_id})")
            self.model = LLM(
                self.model_id,
                #cpu_offload_gb=40,
                max_model_len=7000,
                #device="cuda" if torch.cuda.is_available() else "cpu",
            )
            logger.debug("Model loaded")
            self.first = False
        
        with open(f"{self.model_id}/generation_config.json", "r") as f:
            gen_config = json.load(f)
        
        print(gen_config)   
        
        sampling_params = SamplingParams(
            temperature=gen_config["temperature"] if "temperature" in gen_config else 1.0,
            top_p=gen_config["top_p"] if "top_p" in gen_config else 1.0,
            repetition_penalty=gen_config["repetition_penalty"] if "repetition_penalty" in gen_config else 1.0,
            top_k=gen_config["top_k"] if "top_k" in gen_config else -1,
            max_tokens=1024,
            n=n_iterations,
            seed=random_seed,
        )
        timer = time.time()
        
        responses = self.model.generate(prompts, sampling_params=sampling_params)
        results = [
            [output.text for output in response.outputs] for response in responses
        ]
        log_time(
            message=f"Generating response for {len(prompts):,} ({n_iterations} iteration(s)) prompts",
            start_time=timer,
            logger=logger,
            level="debug",
            extra={"prompts_count": len(prompts)},
        )
        return results

    def _chunk_batch_encoding(self, encoded_inputs, batch_size):
        # Convert to list of dicts (one per item in the batch)
        total = encoded_inputs.input_ids.shape[0]

        for i in range(0, total, batch_size):
            chunk = {k: v[i : i + batch_size] for k, v in encoded_inputs.items()}
            yield BatchEncoding(chunk)
    
    def batch_generate(
        self,
        prompts: List[str],
        n_iterations=1,
        system_prompt: Union[str, None] = None,
        random_seed=42,
        return_prompts=False,
    ) -> List[List[str]] | Tuple[List[List[str]], List[str]]:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        base_chat_template = (
            [{"role": "system", "content": system_prompt}]
            if system_prompt is not None
            else []
        )
        
        if base_chat_template:
            prompts = [
                 re.sub(
                    r"Cutting Knowledge Date:.*?\nToday Date:.*?\n+",  # match the two lines
                    "",
                    self.tokenizer.apply_chat_template(
                        [*base_chat_template, {"role": "user", "content": x}],
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    flags=re.S
                )
                for x in prompts
            ]
        else:
            prompts = [
                 re.sub(
                    r"Cutting Knowledge Date:.*?\nToday Date:.*?\n+",  # match the two lines
                    "",
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": x}],
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                    flags=re.S
                )
                for x in prompts
            ]
        
        #print(prompts[0])
        #time.sleep(10)
        
        prompts_encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        answers = self._batch_generate(
            prompts=prompts_encoded,
            n_iterations=n_iterations,
            random_seed=random_seed,
        )
        return answers if not return_prompts else (answers, prompts)
    
    
    def _batch_generate(
        self,
        prompts: List[str],
        n_iterations=1,
        random_seed: Union[int, None] = None,
    ) -> List[List[str]]:
        if not self.enabled:
            answers = [
                ["The author Llama is disabled" for _ in range(n_iterations)]
                for _ in prompts
            ]
            return answers

        if self.model is None:
            logger.debug(f"Loading Llama model (id {self.model_id})")
            #self.model = AutoModelForCausalLM.from_pretrained(
            #    self.model_id,
            #    device_map="auto",
            #)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                #device_map="cuda",
                attn_implementation="flash_attention_2"
            )
            
            
            ds_engine = deepspeed.init_inference(
                model=model,
                dtype=torch.bfloat16,
                replace_with_kernel_inject=False,
            )

            self.model = ds_engine.module
            
            logger.debug("Model loaded")
            self.first = False

        timer = time.time()
        
        results = []
        
        print("Start generating...")
        
        for batch in self._chunk_batch_encoding(encoded_inputs=prompts, batch_size=100):
            outputs = self.model.generate(
                                **batch,
                                do_sample=False,
                                max_new_tokens=700,
                                temperature=None,
                                top_p=None
                            )
            # Remove prompt from generated output
            outputs = outputs[:, prompts["input_ids"].shape[-1]:]
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            #decoded_output = [re.sub(r'^/gin[>\w]*\s*', '', t) for t in decoded_output]
            #decoded_output = [re.sub(r'^*/gin[>\w]*\s', '', t) for t in decoded_output]

            results.extend(decoded_output)
        
        log_time(
            message=f"Generating response for {len(prompts):,} ({n_iterations} iteration(s)) prompts",
            start_time=timer,
            logger=logger,
            level="debug",
            extra={"prompts_count": len(prompts)},
        )
        return results
