import threading
import time
import logging
import queue
import concurrent.futures
import random
import sys
import logging.handlers
from openai import OpenAI
from typing import Dict, Any, Callable, Optional, Tuple
from zhipuai import ZhipuAI

# Global switch for ZhipuAI usage
USE_ZHIPUAI = False

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger("api_client_manager")
    logger.setLevel(logging.DEBUG)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        'api_client.log', maxBytes=10485760, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

class ApiClientManager:
    """Singleton class for managing OpenAI/ZhipuAI API clients with high reliability"""
    _instance = None
    _lock = threading.Lock()
    
    # Request queue and worker thread pool
    _request_queue = queue.PriorityQueue()
    _worker_threads = []
    _stop_event = threading.Event()
    
    # Multi-client support
    _clients = {}
    _clients_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._start_workers()
        return cls._instance
    
    @classmethod
    def set_use_zhipuai(cls, use_zhipuai=True):
        """Set whether to use ZhipuAI API"""
        global USE_ZHIPUAI
        previous = USE_ZHIPUAI
        USE_ZHIPUAI = use_zhipuai
        logger.info(f"API client manager: ZhipuAI mode switched from {previous} to {USE_ZHIPUAI}")
    
    @classmethod
    def is_using_zhipuai(cls):
        global USE_ZHIPUAI
        return USE_ZHIPUAI
    
    def __init__(self):
        self.client = None
        self.last_request_time = 0
        
        # Request interval control
        self.min_request_interval = 0.05  # 50ms, balance speed and stability
        self.max_interval = 0.5  # Maximum interval
        
        # Thread pool configuration
        self.num_workers = 9
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=24)
        
        # Request monitoring
        self.request_count = 0
        self.error_count = 0
        
        # Maximum retry attempts
        self.max_retries = 3
    
    def _start_workers(self):
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self._worker_threads.append(thread)
        logger.info(f"Started {self.num_workers} worker threads")
    
    def _worker_loop(self, worker_id):
        logger.debug(f"Worker thread {worker_id} started")
        while not self._stop_event.is_set() or not self._request_queue.empty():
            try:
                # Get request
                try:
                    priority, request_id, (request_func, args, kwargs, result_holder) = self._request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process request
                try:
                    # Execute request directly without cache
                    response = self._execute_request_internal(request_func, *args, **kwargs)
                    
                    # Validate response
                    if not self._validate_response(response):
                        raise ValueError(f"API returned invalid response: {response}")
                        
                    result_holder["response"] = response
                    result_holder["error"] = None
                    
                except Exception as e:
                    # Handle retry logic
                    if result_holder.get("retry_count", 0) < self.max_retries:
                        # Increment retry count
                        result_holder["retry_count"] = result_holder.get("retry_count", 0) + 1
                        retry_delay = 0.5 * result_holder["retry_count"]  # Simple linear backoff
                        
                        logger.warning(f"Request failed, retry {result_holder['retry_count']}/{self.max_retries}, "
                                      f"delay {retry_delay:.1f}s: {str(e)}")
                        
                        # Re-queue request
                        time.sleep(retry_delay)
                        self._request_queue.put((priority + 1, request_id, 
                                              (request_func, args, kwargs, result_holder)))
                        self._request_queue.task_done()
                        continue
                    else:
                        # Max retries reached
                        result_holder["error"] = e
                        logger.error(f"Request failed, max retries reached: {str(e)}")
                
                # Notify result
                result_holder["event"].set()
                self._request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker thread {worker_id} error processing request: {str(e)}")
    
    def _validate_response(self, response):
        if response is None:
            return False
            
        try:
            # Validate OpenAI/ZhipuAI format response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    content = response.choices[0].message.content
                    return content is not None and len(content.strip()) > 0
                elif hasattr(response.choices[0], 'text'):
                    content = response.choices[0].text
                    return content is not None and len(content.strip()) > 0
            
            # Validate Spark format response
            if hasattr(response, 'data') and hasattr(response.data, 'text'):
                return response.data.text is not None and len(response.data.text.strip()) > 0
            
            # Other response types
            return True
        except Exception as e:
            logger.warning(f"Response validation failed: {str(e)}")
            return False
    
    def _execute_request_internal(self, request_func, *args, **kwargs):
        # Control request interval
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            # Add small jitter to prevent request synchronization
            jitter = random.uniform(0, 0.01)
            sleep_time = self.min_request_interval - elapsed + jitter
            time.sleep(sleep_time)
        
        # Execute request
        start_time = time.time()
        try:
            response = request_func(*args, **kwargs)
            req_time = time.time() - start_time
            self.last_request_time = time.time()
            self.request_count += 1
            
            logger.debug(f"API request successful, time: {req_time:.2f}s")
            return response
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"API request error, time: {time.time() - start_time:.2f}s, {str(e)}")
            raise
    
    def initialize_client(self, api_key, base_url):
        global USE_ZHIPUAI
        
        if USE_ZHIPUAI:
            client_key = f"zhipuai:{api_key[:8]}"
        else:
            client_key = f"{base_url}:{api_key[:8]}"
        
        with self._clients_lock:
            if client_key not in self._clients:
                try:
                    if USE_ZHIPUAI:
                        client = ZhipuAI(api_key=api_key)
                        logger.info(f"Initialized ZhipuAI client")
                    else:
                        client = OpenAI(
                            api_key=api_key,
                            base_url=base_url,
                            timeout=30.0,  # Reasonable timeout setting
                            max_retries=1,  # Low client-level retry, handled by manager
                        )
                        logger.info(f"Initialized {'ZhipuAI' if USE_ZHIPUAI else 'OpenAI'} API client: {'' if USE_ZHIPUAI else base_url}")
                    
                    self._clients[client_key] = client
                    
                    if self.client is None:
                        self.client = client
                except Exception as e:
                    logger.error(f"Client initialization failed: {str(e)}")
                    raise
            
            return self._clients[client_key]
    
    def get_client(self, api_key, base_url):
        global USE_ZHIPUAI
        
        if USE_ZHIPUAI:
            client_key = f"zhipuai:{api_key[:8]}"
        else:
            client_key = f"{base_url}:{api_key[:8]}"
        
        with self._clients_lock:
            if client_key in self._clients:
                return self._clients[client_key]
        
        return self.initialize_client(api_key, base_url)
    
    def execute_request(self, request_func, *args, priority=5, **kwargs):
        # Remove priority from kwargs if not accepted by request_func
        if 'priority' in kwargs and not any(p.name == 'priority' for p in request_func.__code__.co_varnames):
            priority = kwargs.pop('priority', 5)
        
        # Create result container
        result_holder = {
            "response": None,
            "error": None,
            "event": threading.Event(),
            "retry_count": 0
        }
        
        # Unique request ID
        request_id = time.time() + random.random()
        
        # Add to queue
        self._request_queue.put((priority, request_id, (request_func, args, kwargs, result_holder)))
        
        # Wait for result
        result_holder["event"].wait()
        
        # Check for errors
        if result_holder["error"]:
            raise result_holder["error"]
        
        # Final response validation
        if not self._validate_response(result_holder["response"]):
            raise ValueError("Final response validation failed")
            
        return result_holder["response"]
    
    def execute_request_batch(self, request_funcs_with_args):
        futures = []
        results = []
        
        # Submit all requests in parallel
        for func, args, kwargs in request_funcs_with_args:
            future = self.executor.submit(self.execute_request, func, *args, **kwargs)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(e)
        
        return results
    
    def shutdown(self):
        logger.info("Shutting down API client manager...")
        self._stop_event.set()
        
        # Wait for all threads to complete
        for i, thread in enumerate(self._worker_threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        logger.info("API client manager shutdown complete")