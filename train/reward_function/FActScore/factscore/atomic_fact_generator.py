import json
import os
import asyncio
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Optional, Union

from .openai_lm import OpenAIModel
from .prompts import ATOMIC_FACT_PROMPT
from .utils import detect_initials, fix_sentence_splitter

class AtomicFactGenerator:
    """Atomic fact generator for decomposing text into independent factual statements"""
    
    def __init__(self, 
                 openai_key, 
                 base_url=None,
                 demo_dir=None, 
                 model_name="ChatGPT", 
                 model_version=None,
                 cache_file=None,
                 client_manager=None,
                 num_ex=4,
                 post_process=True,
                 use_async=False,
                 async_concurrent_limit=10,
                 async_batch_size=10):
        """
        Initialize atomic fact generator
        
        Args:
            openai_key: OpenAI API key
            base_url: Custom base URL for OpenAI API (optional)
            demo_dir: Directory containing demo files
            model_name: Model name to use, only 'ChatGPT' is supported
            model_version: Model version
            cache_file: Cache file path
            num_ex: Number of examples to use
            post_process: Whether to apply post-processing
            use_async: Whether to use async processing for batch text processing
            async_concurrent_limit: Maximum concurrent async calls
            async_batch_size: Async batch processing size
        """
        if model_name != "ChatGPT":
            raise ValueError(f"Unsupported model: {model_name}, only ChatGPT is currently supported")
            
        self.demo_dir = demo_dir
        self.num_ex = num_ex
        self.model_name = model_name
        self.post_process = post_process
        self.demos = {}
        self.client_manager = client_manager
        
        # Async processing configuration
        self.use_async = use_async
        self.async_concurrent_limit = async_concurrent_limit
        self.async_batch_size = async_batch_size
        
        # Initialize OpenAI model with async parameters
        self.openai_lm = OpenAIModel(
            model_name, 
            model_version, 
            cache_file=cache_file, 
            openai_key=openai_key,
            base_url=base_url,
            use_async=use_async,
            async_concurrent_limit=async_concurrent_limit,
            async_batch_size=async_batch_size
        )

    def save_cache(self):
        """Save model cache"""
        self.openai_lm.save_cache()

    def run(self, text_or_texts, cost_estimate=None):
        """
        Convert text or list of texts into atomic fact sets
        
        Args:
            text_or_texts: Single text string or list of text strings
            cost_estimate: If not None, returns total word count cost estimation
            
        Returns:
            For single text: list of atomic facts or cost estimate
            For text list: list of atomic fact lists or list of cost estimates
        """
        # Check if input is a list of texts
        if isinstance(text_or_texts, list):
            # Process text list
            if self.use_async:
                # Use async processing
                return self._process_texts_async(text_or_texts, cost_estimate)
            else:
                # Use sync processing
                return self._process_texts_sync(text_or_texts, cost_estimate)
        else:
            # Process single text
            assert isinstance(text_or_texts, str), "text must be a string"
            paragraphs = [para.strip() for para in text_or_texts.split("\n") if len(para.strip()) > 0]
            return self.get_atomic_facts_from_text(paragraphs, cost_estimate=cost_estimate)

    def _process_texts_sync(self, texts, cost_estimate=None):
        """
        Synchronously process list of texts
        
        Args:
            texts: List of text strings
            cost_estimate: Cost estimation
            
        Returns:
            List of atomic fact lists
        """
        # If cost estimation, calculate for each text separately
        if cost_estimate:
            results = []
            for text in texts:
                paragraphs = [para.strip() for para in text.split("\n") if len(para.strip()) > 0]
                cost = self.estimate_cost("\n\n".join(paragraphs), [], cost_estimate)
                results.append(cost)
            return results
            
        # Prepare prompts for all texts
        prompts = []
        
        for text in texts:
            paragraphs = [para.strip() for para in text.split("\n") if len(para.strip()) > 0]
            full_text = "\n\n".join(paragraphs)
            prompt = ATOMIC_FACT_PROMPT.format(text=full_text)
            prompts.append(prompt)
        
        # Process batch requests synchronously
        batch_results = self.openai_lm.generate_batch(
            prompt_batch=prompts,
            chunk_size=self.async_batch_size
        )
        
        # Parse results, return only atomic fact lists
        all_results = []
        
        for output, _ in batch_results:
            # Parse atomic facts
            atomic_facts = self.parse_atomic_facts_response(output)
            # Directly add parsed atomic facts
            all_results.append(atomic_facts)
        
        return all_results

    def _process_texts_async(self, texts, cost_estimate=None):
        """
        Asynchronously process list of texts using async batch processing
        
        Args:
            texts: List of text strings
            cost_estimate: Cost estimation
            
        Returns:
            List of atomic fact lists
        """
        # Async processing needs to run in event loop
        loop = asyncio.get_event_loop()
        try:
            return loop.run_until_complete(self._async_process(texts, cost_estimate))
        except RuntimeError:
            # If event loop is closed, create new event loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(self._async_process(texts, cost_estimate))

    async def _async_process(self, texts, cost_estimate=None):
        """
        Actual async processing logic
        
        Args:
            texts: List of text strings
            cost_estimate: Cost estimation
            
        Returns:
            List of atomic fact lists
        """
        # If cost estimation, calculate for each text separately
        if cost_estimate:
            results = []
            for text in texts:
                paragraphs = [para.strip() for para in text.split("\n") if len(para.strip()) > 0]
                cost = self.estimate_cost("\n\n".join(paragraphs), [], cost_estimate)
                results.append(cost)
            return results
            
        # Prepare prompts for all texts
        prompts = []
        
        for text in texts:
            paragraphs = [para.strip() for para in text.split("\n") if len(para.strip()) > 0]
            full_text = "\n\n".join(paragraphs)
            prompt = ATOMIC_FACT_PROMPT.format(text=full_text)
            prompts.append(prompt)
        
        # Use async batch processing
        generated_texts, _ = await self.openai_lm._generate_batch_async(
            prompt_batch=prompts,
            max_sequence_length=8192,
            max_output_length=8192
        )
        
        # Parse results, return only atomic fact lists
        all_results = []
        
        for output in generated_texts:
            # Parse atomic facts
            atomic_facts = self.parse_atomic_facts_response(output)
            # Directly add parsed atomic facts
            all_results.append(atomic_facts)
        
        return all_results

    def get_atomic_facts_from_text(self, paragraphs, cost_estimate=None):
        """Extract atomic facts directly from text"""
        # Combine all paragraphs into complete text
        full_text = "\n\n".join(paragraphs)
        
        # Only for cost estimation
        if cost_estimate:
            # Calculate and return cost estimate
            words_estimate = self.estimate_cost(full_text, [], cost_estimate)
            return words_estimate
        
        # Get and directly return atomic facts list
        return self.extract_atomic_facts(full_text)

    def estimate_cost(self, full_text, sentences, cost_estimate_type):
        """Estimate API call cost"""
        prompt = ATOMIC_FACT_PROMPT.format(text=full_text)
        
        if cost_estimate_type == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
            return 0
        
        return len(prompt.split())
    
    def extract_atomic_facts(self, text):
        """Extract all atomic facts from complete text in one go"""
        prompt = ATOMIC_FACT_PROMPT.format(text=text)
        output, _ = self.openai_lm.generate(prompt)
        # Parse model output to extract atomic facts
        atomic_facts = self.parse_atomic_facts_response(output)
        return atomic_facts
    
    def is_non_factual_response(self, facts):
        """
        Check if response is non-factual
        
        If facts is a string starting with "NON_FACTUAL:", consider it non-factual response
        """
        # If facts is a string (possibly NON_FACTUAL response)
        if isinstance(facts, str) and facts.startswith("NON_FACTUAL:"):
            return True
        # If facts is a single-item list with special marker
        elif isinstance(facts, list) and len(facts) == 1 and isinstance(facts[0], str) and facts[0].startswith("NON_FACTUAL:"):
            return True
        return False
        
    def parse_atomic_facts_response(self, response):
        """Parse model response to extract atomic facts"""
        # Check if response is None
        if response is None:
            return ["NON_FACTUAL: Failed to get response from model."]
        # Check if content is non-factual
        if "NON_FACTUAL:" in response:
            # Return complete NON_FACTUAL message as special marker
            for line in response.strip().split('\n'):
                if "NON_FACTUAL:" in line:
                    return [line.strip()]
            # If no specific NON_FACTUAL line found, return generic marker
            return ["NON_FACTUAL: This input contains no factual statements to extract."]
        
        # Split output into separate lines
        lines = response.strip().split('\n')
        facts = []
        seen_facts = set()  # For deduplication
        
        # Extract lines starting with dash, these are atomic facts
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                fact = line[2:].strip()  # Remove dash and space
                
                # Filter out atomic facts with less than 3 words
                word_count = len(fact.split())
                if word_count < 3:
                    continue
                
                # Deduplication
                fact_lower = fact.lower()  # Case-insensitive deduplication
                if fact_lower not in seen_facts:
                    facts.append(fact)
                    seen_facts.add(fact_lower)
        
        return facts