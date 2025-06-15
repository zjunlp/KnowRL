from .lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Union

from zhipuai import ZhipuAI

class OpenAIModel(LM):
    def __init__(self, model_name, model_version=None, cache_file=None, read_only_cache=None, openai_key=None, base_url=None, 
                 use_async=False, async_concurrent_limit=10, async_batch_size=10, async_max_retries=5):
        self.model_name = model_name
        self.use_zhipuai = (base_url == "zhipuai")
        
        if model_version is None:
            if model_name == "ChatGPT":
                self.model_version = "gpt-3.5-turbo" if not self.use_zhipuai else "glm-4-flash"
            else:
                raise ValueError(f"Unsupported model: {model_name}, only ChatGPT is currently supported")
        else:
            self.model_version = model_version
            
        self.openai_key = openai_key
        self.base_url = base_url if not self.use_zhipuai else None
        self.temp = 0.0
        self.save_interval = 100
        
        # Async parameters
        self.use_async = use_async
        self.async_concurrent_limit = async_concurrent_limit
        self.async_batch_size = async_batch_size
        self.async_max_retries = async_max_retries
        self.async_client = None
        self.semaphore = None
        self.async_stats = {
            "success": 0,
            "errors": 0,
            "retries": 0,
            "response_times": []
        }
        
        super().__init__(cache_file=cache_file, read_only_cache=read_only_cache)

    def load_model(self):
        assert self.openai_key is not None, "Please provide your API key via openai_key parameter"
        api_key = self.openai_key
        
        if self.use_zhipuai:
            self.client = ZhipuAI(api_key=api_key)
            logging.info(f"Loaded ZhipuAI model: {self.model_version}")
            
            if self.use_async:
                self.async_client = self.client
                self.semaphore = asyncio.Semaphore(self.async_concurrent_limit)
        else:
            if self.base_url:
                self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
                if self.use_async:
                    self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
            else:
                self.client = openai.OpenAI(api_key=api_key)
                if self.use_async:
                    self.async_client = AsyncOpenAI(api_key=api_key)
            
            if self.use_async:
                self.semaphore = asyncio.Semaphore(self.async_concurrent_limit)
            
        self.model = self.model_name
        logging.info(f"Loaded model: {self.model_name} ({self.model_version}) {'[async mode]' if self.use_async else ''}")

    def _generate(self, prompt, max_sequence_length=8192, max_output_length=8192):
        # Save cache periodically
        if self.add_n % self.save_interval == 0:
            self.save_cache()
            
        if self.model_name == "ChatGPT":
            message = [{"role": "user", "content": prompt}]
            
            if self.use_async:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    logging.warning("Already in async environment, calling async method directly")
                    result = loop.create_task(self._generate_async(prompt, max_sequence_length, max_output_length))
                    return result
                else:
                    result = loop.run_until_complete(self._generate_async(prompt, max_sequence_length, max_output_length))
                    return result
            
            if self.use_zhipuai:
                response = call_ZhipuAI(
                    message, 
                    model_name=self.model_version, 
                    max_len=max_output_length,
                    temp=self.temp, 
                    client=self.client
                )
            else:
                response = call_ChatGPT(
                    message, 
                    model_name=self.model_version, 
                    max_len=max_output_length,
                    temp=self.temp, 
                    client=self.client
                )
            
            output = response["choices"][0]["message"]["content"]
            return output, response
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported, only ChatGPT is currently supported")

    async def _generate_async(self, prompt, max_sequence_length=8192, max_output_length=8192):
        if self.model_name == "ChatGPT":
            message = [{"role": "user", "content": prompt}]
            
            async with self.semaphore:
                if self.use_zhipuai:
                    response = await call_ZhipuAI_async(
                        message, 
                        model_name=self.model_version, 
                        max_len=max_output_length,
                        temp=self.temp, 
                        client=self.async_client
                    )
                else:
                    response = await call_ChatGPT_async(
                        message, 
                        model_name=self.model_version, 
                        max_len=max_output_length,
                        temp=self.temp, 
                        client=self.async_client
                    )
                
                output = response["choices"][0]["message"]["content"]
                return output, response
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported, only ChatGPT is currently supported")

    def _generate_batch(self, prompt_batch, max_sequence_length=8192, max_output_length=8192):
        # Use async batch processing if async is enabled
        if self.use_async:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logging.warning("Already in async environment, returning task")
                return loop.create_task(self._generate_batch_async(prompt_batch, max_sequence_length, max_output_length))
            else:
                return loop.run_until_complete(self._generate_batch_async(prompt_batch, max_sequence_length, max_output_length))
                
        if self.model_name == "ChatGPT":
            generated_texts = []
            generated_arrs = []
            
            messages = []
            for prompt in prompt_batch:
                messages.append([{"role": "user", "content": prompt}])
            
            for message in messages:
                if self.use_zhipuai:
                    response = call_ZhipuAI(
                        message,
                        model_name=self.model_version,
                        max_len=max_output_length,
                        temp=self.temp,
                        client=self.client
                    )
                else:
                    response = call_ChatGPT(
                        message,
                        model_name=self.model_version,
                        max_len=max_output_length,
                        temp=self.temp,
                        client=self.client
                    )
                
                output = response["choices"][0]["message"]["content"]
                generated_texts.append(output)
                generated_arrs.append(response)
                
            return generated_texts, generated_arrs
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported, only ChatGPT is currently supported")

    async def _generate_batch_async(self, prompt_batch, max_sequence_length=8192, max_output_length=8192):
        if self.model_name == "ChatGPT":
            all_tasks = []
            
            for prompt in prompt_batch:
                task = asyncio.create_task(
                    self._generate_async(
                        prompt=prompt,
                        max_sequence_length=max_sequence_length,
                        max_output_length=max_output_length
                    )
                )
                all_tasks.append(task)
            
            generated_texts = []
            generated_arrs = []
            
            for i in range(0, len(all_tasks), self.async_batch_size):
                batch_tasks = all_tasks[i:i+self.async_batch_size]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logging.error(f"Async processing error: {result}")
                        generated_texts.append(f"Error: {str(result)}")
                        generated_arrs.append({"error": str(result)})
                    else:
                        output, response = result
                        generated_texts.append(output)
                        generated_arrs.append(response)
            
            return generated_texts, generated_arrs
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported, only ChatGPT is currently supported")
    
    def get_async_stats(self):
        stats = self.async_stats.copy()
        
        if self.async_stats["response_times"]:
            stats["avg_response_time"] = sum(self.async_stats["response_times"]) / len(self.async_stats["response_times"])
            stats["min_response_time"] = min(self.async_stats["response_times"])
            stats["max_response_time"] = max(self.async_stats["response_times"])
        
        return stats


async def call_ZhipuAI_async(message, model_name="glm-4-flash", max_len=8192, temp=0.0, client=None):
    retry_count = 0
    max_retries = 5
    start_time = time.time()
    
    while True:
        try:
            # ZhipuAI currently doesn't have official async API, use sync API
            api_response = client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=max_len,
                temperature=temp
            )
            
            # Convert ZhipuAI response to standard format (OpenAI compatible)
            response = {
                "choices": [{
                    "message": {
                        "content": api_response.choices[0].message.content
                    }
                }],
                "id": api_response.id,
                "model": api_response.model,
                "usage": api_response.usage.model_dump() if hasattr(api_response, "usage") else {},
                "response_time": time.time() - start_time
            }
            
            logging.info(f"ZhipuAI request completed successfully, response time: {response['response_time']:.2f}s")
            return response
            
        except Exception as e:
            retry_count += 1
            error_time = time.time() - start_time
            
            if "BadRequestError" in str(e) or "InvalidRequestError" in str(e):
                logging.critical(f"Invalid request error\nInput prompt:\n\n{message}\n\n")
                raise
            
            if retry_count >= max_retries:
                logging.error(f"ZhipuAI request failed, max retries reached: {e}")
                raise
                
            delay = min(2 ** retry_count, 60)
            logging.warning(f"ZhipuAI API error (time {error_time:.2f}s): {e}. Retry {retry_count}/{max_retries}, waiting {delay}s...")
            
            await asyncio.sleep(delay)


async def call_ChatGPT_async(message, model_name="gpt-3.5-turbo", max_len=8192, temp=0.0, client=None):
    retry_count = 0
    max_retries = 5
    start_time = time.time()
    
    while True:
        try:
            api_response = await client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=max_len,
                temperature=temp
            )
            
            response = {
                "choices": [{
                    "message": {
                        "content": api_response.choices[0].message.content
                    }
                }],
                "id": api_response.id,
                "model": api_response.model,
                "usage": api_response.usage.model_dump() if hasattr(api_response, "usage") else {},
                "response_time": time.time() - start_time
            }
            
            logging.info(f"OpenAI request completed successfully, response time: {response['response_time']:.2f}s")
            return response
            
        except Exception as e:
            retry_count += 1
            error_time = time.time() - start_time
            
            if "BadRequestError" in str(e) or "InvalidRequestError" in str(e):
                logging.critical(f"Invalid request error\nInput prompt:\n\n{message}\n\n")
                raise
            
            if retry_count >= max_retries:
                logging.error(f"OpenAI request failed, max retries reached: {e}")
                raise
                
            delay = min(2 ** retry_count, 60)
            logging.warning(f"OpenAI API error (time {error_time:.2f}s): {e}. Retry {retry_count}/{max_retries}, waiting {delay}s...")
            
            await asyncio.sleep(delay)


def call_ZhipuAI(message, model_name="glm-4-flash", max_len=8192, temp=0.0, verbose=False, client=None):
    response = None
    received = False
    num_rate_errors = 0
    
    while not received:
        try:
            api_response = client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=max_len,
                temperature=temp
            )
            
            # Convert ZhipuAI response to standard format (OpenAI compatible)
            response = {
                "choices": [{
                    "message": {
                        "content": api_response.choices[0].message.content
                    }
                }],
                "id": api_response.id,
                "model": api_response.model,
                "usage": api_response.usage.model_dump() if hasattr(api_response, "usage") else {}
            }
            received = True
            
        except Exception as e:
            num_rate_errors += 1
            error = sys.exc_info()[0]
            
            if "BadRequestError" in str(e) or "InvalidRequestError" in str(e):
                logging.critical(f"Invalid request error\nInput prompt:\n\n{message}\n\n")
                assert False
            
            logging.error(f"ZhipuAI API error: {error} ({num_rate_errors}). Waiting {np.power(2, num_rate_errors):.2f} seconds")
            time.sleep(np.power(2, num_rate_errors))
            
    return response

def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=8192, temp=0.0, verbose=False, client=None):
    response = None
    received = False
    num_rate_errors = 0
    
    while not received:
        try:
            api_response = client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=max_len,
                temperature=temp
            )
            
            response = {
                "choices": [{
                    "message": {
                        "content": api_response.choices[0].message.content
                    }
                }],
                "id": api_response.id,
                "model": api_response.model,
                "usage": api_response.usage.model_dump() if hasattr(api_response, "usage") else {}
            }
            received = True
            
        except Exception as e:
            num_rate_errors += 1
            error = sys.exc_info()[0]
            
            if "BadRequestError" in str(e) or "InvalidRequestError" in str(e):
                logging.critical(f"Invalid request error\nInput prompt:\n\n{message}\n\n")
                assert False
            
            logging.error(f"API error: {error} ({num_rate_errors}). Waiting {np.power(2, num_rate_errors):.2f} seconds")
            time.sleep(np.power(2, num_rate_errors))
            
    return response