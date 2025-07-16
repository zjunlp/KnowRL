import argparse
import string
import json
import numpy as np
import os
import logging
import time
from collections import defaultdict
from itertools import islice

from tqdm import tqdm
from .abstain_detection import is_response_abstained
from .atomic_fact_generator import AtomicFactGenerator
from .openai_lm import OpenAIModel
from .retrieval import DocDB, Retrieval

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk

class FactScorer:
    """Fact scorer class for evaluating factual accuracy of generated text"""

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        openai_key=None,
        base_url=None,
        cost_estimate="consider_cache",
        abstain_detection_type=None,
        batch_size=512,
        client_manager=None,
        af_model_name="ChatGPT",
        af_model_version=None,
        use_nli=False,
        nli_model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        nli_entailment_threshold=0.3,
        quantization_type=None,
        verbose=False,
        device=None,
        dedicated_gpu_id=None,
    ):
        """
        Initialize fact scorer

        Args:
            data_dir: Data directory path
            cache_dir: Cache directory path
            openai_key: OpenAI API key
            base_url: Custom base URL for OpenAI API (optional)
            cost_estimate: Cost estimation method
            abstain_detection_type: Answer abstention detection type
            batch_size: Batch processing size
            af_model_name: Atomic fact generator model name
            af_model_version: Atomic fact generator model version
            client_manager: API client manager (optional)
            use_nli: Whether to use NLI model for atomic fact judgment
            nli_model_name: NLI model name
            nli_entailment_threshold: NLI judgment threshold
            quantization_type: Model quantization type, options: None, "dynamic", "static", "int8"
            verbose: Whether to output detailed information
            device: Device to use, e.g., "cuda:0", "cuda:1" or "cpu" (default None for auto-selection)
        """
        self.model_name = "retrieval+ChatGPT"
        self.verbose = verbose
        if self.verbose:
            print(f"Model name: {self.model_name}")

        # Cache control parameters
        self.sentence_cache = {}
        self.nli_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Data and cache directory settings
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.db = {}
        self.retrieval = {}
        self.batch_size = batch_size
        self.openai_key = openai_key
        self.base_url = base_url
        self.af_model_name = af_model_name
        self.af_model_version = af_model_version
        self.abstain_detection_type = abstain_detection_type
        self.af_generator = None
        self.cost_estimate = cost_estimate
        self.client_manager = client_manager
        
        # Database name to path mapping
        self.db_paths = {}
        
        # NLI model related parameters
        self.use_nli = use_nli
        self.nli_model_name = nli_model_name
        self.nli_entailment_threshold = nli_entailment_threshold
        self.nli_pipeline = None
        self.optimized_model = None
        self.nli_batch_size = 32
        self.quantization_type = quantization_type

        # GPU resource management
        self.dedicated_gpu_id = dedicated_gpu_id
        
        if dedicated_gpu_id is not None and torch.cuda.is_available():
            if torch.cuda.device_count() > dedicated_gpu_id:
                print(f"Using GPU {dedicated_gpu_id}")
                torch.cuda.set_device(dedicated_gpu_id)
                self.device = torch.device(f"cuda:{dedicated_gpu_id}")
        
        # Device setting
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize ChatGPT model
        chatgpt_cache_path = self._get_cache_path("ChatGPT.pkl")
        self.lm = OpenAIModel(
            model_name=self.af_model_name,
            model_version=self.af_model_version,
            cache_file=chatgpt_cache_path,
            openai_key=openai_key,
            base_url=base_url,
        )
        
        # Download NLTK punkt tokenizer data if using NLI
        if self.use_nli:
            try:
                nltk.data.find('tokenizers/punkt')
            except nltk.downloader.DownloadError:
                logging.info("NLTK 'punkt' tokenizer data not found. Downloading...")
                try:
                    nltk.download('punkt', quiet=True)
                    logging.info("'punkt' data downloaded successfully.")
                except Exception as e:
                    logging.error(f"Failed to download 'punkt' data: {e}. Sentence splitting might be suboptimal.")
            
            self._optimize_nli_model()

    def _quantize_model(self, model=None):
        """
        Apply quantization to model for memory reduction and inference speedup
        
        Note: Dynamic quantization in PyTorch currently only supports CPU backend, not CUDA
        
        Args:
            model: Model to quantize, if None uses self.optimized_model
            
        Returns:
            Quantized model (returns non-quantized model if running on GPU)
        """
        if model is None:
            if not hasattr(self, 'optimized_model') or self.optimized_model is None:
                return None
            model = self.optimized_model
            
        if self.quantization_type is None:
            return model
        
        device_to_use = self.device
    
        if device_to_use.type == 'cuda':
            logging.warning("Quantized models do not support CUDA backend. Using original non-quantized model on GPU.")
            return model
            
        try:
            model = model.cpu()
            
            logging.info(f"Applying {self.quantization_type} quantization (CPU mode only)...")
            
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            logging.info("Applied dynamic quantization (Int8)")
            
            return quantized_model
            
        except Exception as e:
            logging.error(f"Model quantization failed: {e}")
            logging.warning("Continuing with original non-quantized model")
            return model

    def _optimize_nli_model(self):
        """Optimize NLI model initialization and configuration"""
        if self.optimized_model is not None:
            return True
                
        try:
            logging.info(f"Initializing and optimizing NLI model: {self.nli_model_name}")
            
            if not hasattr(self, 'nli_cache'):
                self.nli_cache = {}
            
            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            
            # GPU optimizations
            if self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
                
                if torch.cuda.get_device_capability(self.device.index)[0] >= 7:
                    model = model.half()
                    logging.info("Using FP16 acceleration for NLI model")
            
            model.config.use_cache = True
            model = model.to(self.device)
            model.eval()
            
            # Set labels
            try:
                if hasattr(model.config, 'label2id') and model.config.label2id:
                    self.entailment_label = next(label for label, id_ in model.config.label2id.items() 
                                            if label.upper() == 'ENTAILMENT')
                else:
                    self.entailment_label = 'ENTAILMENT'
            except:
                self.entailment_label = 'ENTAILMENT'
            
            self.optimized_model = model
            self.optimized_tokenizer = tokenizer
            
            logging.info(f"NLI model optimization completed, using device: {self.device}")
            return True
    
        except Exception as e:
            logging.error(f"Failed to optimize NLI model: {e}")
            return False

    def _initialize_nli_pipeline(self, device=None):
        """Initialize NLI model and tokenizer into pipeline"""
        if self.nli_pipeline is not None:
            if hasattr(self.nli_pipeline.model, 'name_or_path') and self.nli_pipeline.model.name_or_path == self.nli_model_name:
                return self.nli_pipeline
            else:
                logging.info(f"Switching NLI model to '{self.nli_model_name}', reinitializing pipeline")

        device_to_use = device if device is not None else self.device
        
        if device_to_use.type == 'cuda':
            pipeline_device = device_to_use.index
        else:
            pipeline_device = -1
                
        try:
            logging.info(f"Initializing NLI pipeline with model '{self.nli_model_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            
            if device_to_use.type == 'cpu' and self.quantization_type:
                model = self._quantize_model(model)
                logging.info(f"Applied {self.quantization_type} quantization on CPU")
            elif device_to_use.type == 'cuda' and self.quantization_type:
                logging.warning(f"Quantization not supported on CUDA device, using non-quantized model")
                if torch.cuda.get_device_capability(device_to_use.index)[0] >= 7:
                    model = model.half()
                    logging.info("Using FP16 acceleration")
            
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device
            )
            logging.info(f"NLI pipeline initialized successfully, using device: {device_to_use}")
            return self.nli_pipeline
        except Exception as e:
            logging.error(f"Failed to initialize NLI pipeline: {e}")
            self.nli_pipeline = None
            raise

    def _is_fact_supported_by_text_nli_optimized(self, atomic_fact, passage_text, use_sentence_splitting=False, batch_size=None):
        """
        Optimized NLI judgment function
        
        Optimizations:
        1. Use torch.no_grad() to avoid gradient computation
        2. Use caching mechanism
        3. Automatic mixed precision
        4. Optimized batch processing
        5. Optimized memory management
        """
        if not atomic_fact or not passage_text:
            return False, 0.0, None
            
        cache_key = f"{hash(atomic_fact)}_{hash(passage_text)}"
        if hasattr(self, 'nli_cache') and cache_key in self.nli_cache:
            return self.nli_cache[cache_key]
        
        batch_size = batch_size or getattr(self, 'nli_batch_size', 32)
        
        # Sentence processing
        if use_sentence_splitting:
            try:
                sentences = nltk.sent_tokenize(passage_text)
                if not sentences:
                    sentences = [passage_text]
            except:
                sentences = [s.strip() for s in re.split(r'[.?!]\s+', passage_text) if s.strip()]
                if not sentences:
                    sentences = [passage_text]
        else:
            sentences = [passage_text]
        
        # Batch processing
        texts = sentences
        text_pairs = [atomic_fact] * len(sentences)
        
        # Optimize inference process
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_pairs = text_pairs[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.optimized_tokenizer(batch_texts, batch_pairs, 
                                            return_tensors="pt", 
                                            padding=True, 
                                            truncation=True,
                                            max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast():
                    outputs = self.optimized_model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs)
                
                del outputs
                del inputs
        
        if not all_probs:
            result = (False, 0.0, None)
            if hasattr(self, 'nli_cache'):
                self.nli_cache[cache_key] = result
            return result
        
        all_probs = torch.cat(all_probs, dim=0)
        
        # Get entailment label ID
        entailment_id = None
        for label_id, label_name in self.optimized_model.config.id2label.items():
            if label_name == self.entailment_label:
                entailment_id = int(label_id)
                break
        
        if entailment_id is None:
            result = (False, 0.0, None)
            if hasattr(self, 'nli_cache'):
                self.nli_cache[cache_key] = result
            return result
        
        entailment_probs = all_probs[:, entailment_id].cpu().numpy()
        
        max_prob_idx = np.argmax(entailment_probs)
        max_prob = entailment_probs[max_prob_idx]
        
        is_supported = max_prob >= self.nli_entailment_threshold
        supporting_sentence = sentences[max_prob_idx] if is_supported else None
        
        result = (is_supported, float(max_prob), supporting_sentence)
        if hasattr(self, 'nli_cache'):
            self.nli_cache[cache_key] = result
        
        return result

    def _is_fact_supported_by_text_nli(self, atomic_fact, passage_text, use_sentence_splitting=False, device=None):
        """
        Use NLI model to determine if an atomic fact is supported by retrieved text
        
        Args:
            atomic_fact: Atomic fact to judge (short text)
            passage_text: Retrieved text that may contain supporting information
            use_sentence_splitting: Whether to split text into sentences for individual judgment
            device: Specify device to run model
            
        Returns:
            Tuple (is_supported, max_entailment_score, supporting_sentence)
        """
        if not atomic_fact or not passage_text:
            return False, 0.0, None

        # Use optimized method if available
        if self.optimized_model is not None:
            return self._is_fact_supported_by_text_nli_optimized(atomic_fact, passage_text)

        try:
            nli_pipeline = self._initialize_nli_pipeline(device)
            if nli_pipeline is None:
                return False, 0.0, None
        except Exception as e:
            logging.error(f"Pipeline initialization error: {e}")
            return False, 0.0, None
            
        # Get model configuration to determine label names
        try:
            if hasattr(nli_pipeline.model.config, 'label2id') and nli_pipeline.model.config.label2id:
                entailment_label = next(label for label, id_ in nli_pipeline.model.config.label2id.items() if label.upper() == 'ENTAILMENT')
            else:
                entailment_label = 'ENTAILMENT'
        except:
            entailment_label = 'ENTAILMENT'

        max_score = 0.0
        supporting_sentence_found = None
        is_supported = False

        if use_sentence_splitting:
            try:
                sentences = nltk.sent_tokenize(passage_text)
                if not sentences:
                    return False, 0.0, None
            except Exception as e:
                logging.warning(f"Sentence splitting error: {e}")
                import re
                sentences = [s.strip() for s in re.split(r'[.?!]\s+', passage_text) if s.strip()]
                if not sentences:
                    return False, 0.0, None
                
            input_pairs = [{"text": sentence, "text_pair": atomic_fact} for sentence in sentences]
        else:
            input_pairs = [{"text": passage_text, "text_pair": atomic_fact}]

        try:
            results = nli_pipeline(input_pairs, truncation=True, return_all_scores=True)

            for i, result_list in enumerate(results):
                if not isinstance(result_list, list):
                    continue
                     
                current_sentence = input_pairs[i]['text']

                found_entailment_score = 0.0
                predicted_label = None
                max_label_score = 0.0

                for score_entry in result_list:
                    label = score_entry.get('label')
                    score = score_entry.get('score')
                    if label is not None and score is not None:
                        if label == entailment_label:
                            found_entailment_score = score
                        if score > max_label_score:
                            max_label_score = score
                            predicted_label = label

                if predicted_label is None:
                    continue

                max_score = max(max_score, found_entailment_score)

                if predicted_label == entailment_label:
                    passes_threshold = (self.nli_entailment_threshold is None) or (found_entailment_score >= self.nli_entailment_threshold)
                    if passes_threshold and not is_supported:
                        is_supported = True
                        supporting_sentence_found = current_sentence
                        break

        except Exception as e:
            logging.error(f"NLI prediction error: {e}")
            return False, max_score, supporting_sentence_found

        return is_supported, max_score, supporting_sentence_found

    def _get_cache_path(self, filename):
        """Get cache file path"""
        if self.cache_dir:
            return os.path.join(self.cache_dir, filename)
        return None

    def _get_db_dir(self, db_path):
        """Get database directory"""
        return os.path.dirname(os.path.abspath(db_path))

    def _extract_db_name(self, db_path):
        """Extract database name from path (without extension)"""
        base_name = os.path.basename(db_path)
        return os.path.splitext(base_name)[0]
        
    def save_cache(self):
        """Save all caches"""
        if self.lm:
            self.lm.save_cache()
        
        for k, v in self.retrieval.items():
            v.save_cache()
        
        if self.af_generator:
            self.af_generator.save_cache()

    def register_knowledge_source(self, name=None, db_path=None):
        """
        Register knowledge source
        
        Args:
            name: Knowledge source name (optional)
            db_path: Database path (required)
        """
        if not db_path:
            raise ValueError("Must provide database path (db_path)")
            
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file does not exist: {db_path}")
            
        if not name:
            name = self._extract_db_name(db_path)
            
        if name in self.retrieval:
            print(f"Knowledge source '{name}' already registered, will reload")
            
        self.db_paths[name] = db_path
        
        db_dir = self._get_db_dir(db_path)
        cache_dir = self.cache_dir

        cache_path = os.path.join(cache_dir, f"retrieval-{name}.json") if cache_dir else None
        embed_cache_path = os.path.join(cache_dir, f"retrieval-{name}.pkl") if cache_dir else None

        self.retrieval[name] = Retrieval(
            db_path=db_path, 
            cache_path=cache_path, 
            embed_cache_path=embed_cache_path,
            retrieval_type="gtr-t5-large",
            batch_size=self.batch_size
        )
            
        print(f"Knowledge source '{name}' registered successfully")
        return name

    def get_score(
        self,
        topics,
        generations,
        gamma=10,
        atomic_facts=None,
        knowledge_source=None,
        db_path=None,
        verbose=False,
        chunk_size=0,
        af_model_name=None,
        num_ex=4,
        count_supported=False,
        use_nli=None,
        nli_model_name=None,
        nli_entailment_threshold=None,
        quantization_type=None,
        max_parallel_generations=4,
        use_parallel=False,
        use_async_af_generation=False,
        async_concurrent_limit=10,
        async_batch_size=5,
    ):
        """Get fact score for generated text with optimized memory management"""
        try:
            self.verbose = verbose
            start_time = time.time()
            
            # Update NLI related parameters if provided
            if use_nli is not None:
                self.use_nli = use_nli
                if self.use_nli:
                    print(f"Using NLI model for fact verification")
                    
            if nli_model_name is not None:
                self.nli_model_name = nli_model_name
                if self.use_nli:
                    print(f"NLI model name: {self.nli_model_name}")
                    
            if nli_entailment_threshold is not None:
                self.nli_entailment_threshold = nli_entailment_threshold
                if self.use_nli:
                    print(f"NLI judgment threshold: {self.nli_entailment_threshold}")

            # Process knowledge source
            if not knowledge_source:
                if db_path:
                    knowledge_source = self.register_knowledge_source(db_path=db_path)
                elif len(self.retrieval) == 1:
                    knowledge_source = list(self.retrieval.keys())[0]
                elif "enwiki-20230401" in self.retrieval:
                    knowledge_source = "enwiki-20230401"
                else:
                    raise ValueError("Must provide knowledge_source name or db_path")
            
            if knowledge_source not in self.retrieval:
                raise ValueError(f"Knowledge source '{knowledge_source}' not registered. Please use register_knowledge_source()")
            
            # Process input format
            if isinstance(topics, str) and isinstance(generations, str):
                topics = [topics]
                generations = [generations]
            else:
                assert isinstance(topics, list) and isinstance(generations, list), "`topics` and `generations` should be lists"
                assert len(topics) == len(generations), "`topics` and `generations` should have same length"

            # Set atomic fact model name
            if af_model_name:
                self.af_model_name = af_model_name
            print(f"Atomic fact model name: {self.af_model_name}")

            # Process atomic facts
            fact_gen_time = 0
            if atomic_facts is not None:
                assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have same length"
            else:
                if self.af_generator is None:
                    db_dir = self._get_db_dir(self.db_paths[knowledge_source])
                    cache_dir = self.cache_dir
                    data_dir = self.data_dir if self.data_dir else db_dir

                    cache_file = os.path.join(cache_dir, f"{self.af_model_name}.pkl") if cache_dir else None
                    demo_dir = os.path.join(data_dir, "demos")

                    if not os.path.exists(demo_dir) and data_dir != db_dir:
                        alt_demo_dir = os.path.join(db_dir, "demos")
                        if os.path.exists(alt_demo_dir):
                            demo_dir = alt_demo_dir
                    
                    self.af_generator = AtomicFactGenerator(
                        openai_key=self.openai_key,
                        base_url=self.base_url,
                        demo_dir=demo_dir,
                        model_name=self.af_model_name,
                        model_version=self.af_model_version,
                        cache_file=cache_file,
                        num_ex=num_ex,
                        post_process=True,
                        use_async=use_async_af_generation,
                        async_concurrent_limit=async_concurrent_limit,
                        async_batch_size=async_batch_size
                    )

                # Generate atomic facts - batch processing
                atomic_start_time = time.time()
                
                # Filter non-abstained responses
                valid_generations = []
                valid_indices = []
                
                for i, gen in enumerate(generations):
                    response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                    if not response_abstained:
                        valid_generations.append(gen)
                        valid_indices.append(i)
                
                # Initialize atomic facts list
                atomic_facts = [None] * len(generations)
                
                if valid_generations:
                    if self.verbose:
                        print(f"Batch processing {len(valid_generations)} valid responses to generate atomic facts...")
                        
                    all_afs = self.af_generator.run(valid_generations)
                    
                    # Process results
                    for idx, (i, facts) in enumerate(zip(valid_indices, all_afs)):
                        if not facts or len(facts) == 0:
                            atomic_facts[i] = None
                        elif len(facts) == 1 and facts[0].startswith("NON_FACTUAL:"):
                            atomic_facts[i] = None
                        else:
                            filtered_facts = [fact for fact in facts if not fact.startswith("NON_FACTUAL:")]
                            atomic_facts[i] = filtered_facts if filtered_facts else None
                            
                self.af_generator.save_cache()

                fact_gen_time = time.time() - atomic_start_time
                print(f"Atomic fact generation time: {fact_gen_time:.4f}s")
            
            respond_ratio = np.mean([facts is not None for facts in atomic_facts])

            # Initialize result list and time counters
            out_list = []
            total_time_dict = {
                "fact_gen_time": fact_gen_time, 
                "retrieval_time": 0, 
                "generation_time": 0,
                "nli_time": 0,
                "nli_facts_count": 0
            }
            
            # Pre-load NLI model if using NLI
            if self.use_nli:
                self._optimize_nli_model()
            
            # Process each sample
            if self.verbose:
                topics_iter = tqdm(list(zip(topics, generations, atomic_facts)), desc="Evaluating factual accuracy")
            else:
                topics_iter = zip(topics, generations, atomic_facts)
                
            for i, (topic, generation, facts) in enumerate(topics_iter):
                if facts is None:
                    decisions = []
                    score = 0
                    num_facts = 0
                else:
                    decision, time_dict = self._get_score(topic, generation, facts, knowledge_source)
                    decisions = decision
                    num_facts = len(facts)
                    
                    # Update time statistics
                    for key in time_dict:
                        if key in total_time_dict:
                            total_time_dict[key] += time_dict[key]
                            
                    if self.verbose:
                        print(f"Sample {i} time stats:", time_dict)

                    # Calculate score
                    if count_supported and decisions:
                        score = sum([d["is_supported"] for d in decisions])
                    elif decisions:
                        score = np.mean([d["is_supported"] for d in decisions])
                        
                        # Apply length penalty
                        if gamma:
                            init_score = score
                            penalty = 1.0 if len(facts) > gamma else np.exp(1 - gamma / len(facts))
                            score = penalty * score
                    else:
                        score = 0
                
                # Build result dictionary
                out = {
                    "topic": topic,
                    "generation": generation,
                    "score": score,
                    "respond_ratio": 1.0 if facts is not None else 0.0,
                    "decisions": [decisions],
                    "num_facts_per_response": num_facts,
                    "using_count": count_supported,
                    "using_nli": self.use_nli
                }
                    
                if gamma and facts is not None and not count_supported and decisions:
                    out["init_score"] = np.mean([d["is_supported"] for d in decisions])
                        
                out_list.append(out)
                    
            # Save cache and print time statistics
            self.save_cache()
            print("Total time statistics:", total_time_dict)
            
            return out_list[0] if len(out_list) == 1 else out_list
            
        except Exception as e:
            logging.error(f"Scoring error: {e}")
            raise

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        """
        Get score for single sample
        
        Args:
            topic: Topic
            generation: Generated text
            atomic_facts: List of atomic facts
            knowledge_source: Knowledge source name
            cost_estimate: Cost estimation method
            
        Returns:
            Decision list and time statistics
        """
        if atomic_facts is None or len(atomic_facts) == 0:
            return [], {"retrieval_time": 0, "generation_time": 0, "nli_time": 0, "nli_facts_count": 0}
    
        # Initialize statistics
        decisions = []
        total_words = 0
        retrieval_total_time = 0
        generation_total_time = 0
        nli_total_time = 0
        
        # Get retrieval passages for each atomic fact
        passages_list = []
        for atom in atomic_facts:
            atom = atom.strip()
            if not atom or atom.startswith("NON_FACTUAL:"):
                continue
                
            retrieval_time_start = time.time()
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            retrieval_time = time.time() - retrieval_time_start
            retrieval_total_time += retrieval_time
            
            passages_list.append(passages)
            
        # Filter valid atomic facts and corresponding passages
        valid_facts = []
        valid_passages_list = []
        
        for i, atom in enumerate(atomic_facts):
            atom = atom.strip()
            if atom and not atom.startswith("NON_FACTUAL:") and i < len(passages_list):
                valid_facts.append(atom)
                valid_passages_list.append(passages_list[i])
        
        # Process cost estimation mode
        if cost_estimate:
            for atom, passages in zip(valid_facts, valid_passages_list):
                for psg in passages:
                    definition = f"""I need to verify if the following statement is supported by the provided context about {topic}. 
                    Please respond with 'True' if the statement is fully supported by the context, or 'False' if it's not supported or contradicted.

                    ### CONTEXT:
                    """
                    context = ""
                    for psg in passages:
                        context += f"[{psg['title']}]\n{psg['text'].replace('<s>', '').replace('</s>', '')}\n\n"
                    definition += context.strip()

                    prompt = f"""{definition}

                    ### STATEMENT TO VERIFY:
                    {atom.strip()}

                    ### ANSWER:
                    Based solely on the provided context (not on any external knowledge), the statement is:"""
                    
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
            
            return total_words
        
        # Use NLI model for judgment
        if self.use_nli:
            nli_time_start = time.time()
            
            for i, (atom, passages) in enumerate(zip(valid_facts, valid_passages_list)):
                passage_texts = [passage['text'].replace('<s>', '').replace('</s>', '') for passage in passages]
                
                is_supported_overall = False
                max_confidence_score = 0.0
                best_supporting_sentence = None
                
                for passage_text in passage_texts:
                    is_supported, confidence_score, supporting_sentence = self._is_fact_supported_by_text_nli_optimized(
                        atom, passage_text, use_sentence_splitting=False)
                    
                    if is_supported:
                        is_supported_overall = True
                        
                        if confidence_score > max_confidence_score:
                            max_confidence_score = confidence_score
                            best_supporting_sentence = supporting_sentence
                            
                        break
                
                decisions.append({
                    "atom": atom, 
                    "is_supported": is_supported_overall,
                    "confidence_score": max_confidence_score,
                    "supporting_sentence": best_supporting_sentence,
                    "passages": passages
                })
            
            nli_time = time.time() - nli_time_start
            nli_total_time = nli_time
            
        else:
            # Use LLM to judge each atomic fact
            for atom, passages in zip(valid_facts, valid_passages_list):
                definition = f"""I need to verify if the following statement is supported by the provided context about {topic}. 
                Please respond with 'True' if the statement is fully supported by the context, or 'False' if it's not supported or contradicted.

                ### CONTEXT:
                """
                context = ""
                for psg in passages:
                    context += f"[{psg['title']}]\n{psg['text'].replace('<s>', '').replace('</s>', '')}\n\n"
                definition += context.strip()

                prompt = f"""{definition}

                ### STATEMENT TO VERIFY:
                {atom.strip()}

                ### ANSWER:
                Based solely on the provided context (not on any external knowledge), the statement is:"""
                
                generation_time_start = time.time()
                output = self.lm.generate(prompt)
                generation_time = time.time() - generation_time_start
                generation_total_time += generation_time
                
                # Process output
                generated_answer = output[0].lower()
                if "true" in generated_answer or "false" in generated_answer:
                    if "true" in generated_answer and "false" not in generated_answer:
                        is_supported = True
                    elif "false" in generated_answer and "true" not in generated_answer:
                        is_supported = False
                    else:
                        is_supported = generated_answer.index("true") < generated_answer.index("false")
                else:
                    negative_keywords = ["not", "cannot", "unknown", "information"]
                    is_supported = all([
                        keyword not in generated_answer.lower().translate(
                            str.maketrans("", "", string.punctuation)
                        ).split() 
                        for keyword in negative_keywords
                    ])

                decisions.append({
                    "atom": atom, 
                    "is_supported": is_supported,
                    "passages": passages
                })
        
        return decisions, {
            "retrieval_time": retrieval_total_time, 
            "generation_time": generation_total_time,
            "nli_time": nli_total_time,
            "nli_facts_count": len(valid_facts) if self.use_nli else 0
        }

    def _get_score_batched(self, topic, generation, atomic_facts, knowledge_source, chunk_size=100):
        """Batched score getting (simple delegation form)"""
        return self._get_score(topic, generation, atomic_facts, knowledge_source)

def main():
    parser = argparse.ArgumentParser(description="FactScore - Evaluate factual accuracy of generated text")
    
    parser.add_argument('--input_path',
                      type=str,
                      default="data/labeled/evaluation_data.jsonl",
                      help="Input data path")
    
    parser.add_argument('--db_path',
                      type=str,
                      default=None,
                      help="Knowledge base database path")
                      
    parser.add_argument('--gamma',
                      type=int,
                      default=10,
                      help="Length penalty hyperparameter")
    
    parser.add_argument('--openai_key',
                      type=str,
                      required=True,
                      help="OpenAI API key")

    parser.add_argument('--base_url',
                      type=str,
                      default=None,
                      help="Custom base URL for OpenAI API (optional)")
    
    parser.add_argument('--data_dir',
                      type=str,
                      default=None,
                      help="Data directory path")
    
    parser.add_argument('--cache_dir',
                      type=str,
                      default=None,
                      help="Cache directory path")
    
    parser.add_argument('--cost_estimate',
                      type=str,
                      default="consider_cache",
                      choices=["consider_cache", "ignore_cache"],
                      help="Cost estimation method")
    
    parser.add_argument('--abstain_detection_type',
                      type=str,
                      default=None,
                      choices=["perplexity_ai", "generic", "none"],
                      help="Answer abstention detection type")
    
    parser.add_argument('--use_atomic_facts',
                      action="store_true",
                      help="Use pre-generated atomic facts")
    
    parser.add_argument('--verbose',
                      action="store_true",
                      help="Show detailed output and progress bars")
    
    parser.add_argument('--print_rate_limit_error',
                      action="store_true",
                      help="Print OpenAI API rate limit errors")
    
    parser.add_argument('--n_samples',
                      type=int,
                      default=None,
                      help="Number of samples to process")
    
    parser.add_argument('--count_supported',
                      action="store_true",
                      help="Use count of supported atomic facts rather than ratio as score")
    
    # NLI related parameters
    parser.add_argument('--use_nli',
                      action="store_true",
                      help="Use NLI model instead of LLM to judge if atomic facts are supported")
    
    parser.add_argument('--nli_model_name',
                      type=str,
                      default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                      help="NLI model name")
    
    parser.add_argument('--nli_entailment_threshold',
                      type=float,
                      default=0.5,
                      help="NLI model judgment threshold")
                      
    # Parallel processing parameters
    parser.add_argument('--max_parallel_generations',
                      type=int,
                      default=4,
                      help="Maximum number of generations to process in parallel")

    parser.add_argument('--use_parallel',
                      action="store_true",
                      help="Whether to use parallel processing for multiple generations (for NLI mode)")
                      
    # Model quantization parameters
    parser.add_argument('--quantization_type',
                    type=str,
                    default=None,
                    choices=[None, "dynamic", "static", "int8"],
                    help="NLI model quantization type for memory reduction and CPU inference speedup (Note: only effective in CPU mode)")
        
    parser.add_argument('--device',
                      type=str,
                      default=None,
                      help="Specify device, e.g., 'cuda:0', 'cuda:1' or 'cpu'")
    
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL
    )

    # Create fact scorer
    fs = FactScorer(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        openai_key=args.openai_key,
        base_url=args.base_url,
        cost_estimate=args.cost_estimate,
        abstain_detection_type=args.abstain_detection_type,
        use_nli=args.use_nli,
        nli_model_name=args.nli_model_name,
        nli_entailment_threshold=args.nli_entailment_threshold,
        quantization_type=args.quantization_type,
        verbose=args.verbose,
        device=args.device
    )

    # Register knowledge source if database path is specified
    if args.db_path:
        fs.register_knowledge_source(db_path=args.db_path)

    # Load input data
    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can only specify `--use_atomic_facts` when atomic facts are already in input data"
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
                
            if args.n_samples is not None and tot == args.n_samples:
                break
                
    # Get scores - force disable parallel processing
    results = fs.get_score(
        topics=topics,
        generations=generations,
        gamma=args.gamma,
        atomic_facts=atomic_facts if args.use_atomic_facts else None,
        db_path=args.db_path,
        verbose=args.verbose,
        count_supported=args.count_supported,
        use_nli=args.use_nli,
        nli_model_name=args.nli_model_name,
        nli_entailment_threshold=args.nli_entailment_threshold,
        quantization_type=args.quantization_type,
        max_parallel_generations=args.max_parallel_generations,
        use_parallel=False
    )
    
    # Process result output
    if isinstance(results, list):
        for i, result in enumerate(results):
            if args.count_supported:
                logging.critical(f"Generation {i+1} FActScore (supported fact count) = {result['score']:.1f}")
            else:
                logging.critical(f"Generation {i+1} FActScore = {100 * result['score']:.1f}%")
                if "init_score" in result:
                    logging.critical(f"Generation {i+1} FActScore (no length penalty) = {100 * result['init_score']:.1f}%")
            logging.critical(f"Generation {i+1} atomic fact count = {result['num_facts_per_response']:.1f}")
            logging.critical(f"Generation {i+1} method: {'NLI' if result['using_nli'] else 'LLM'}")
    else:
        result = results
        if args.count_supported:
            logging.critical(f"FActScore (supported fact count) = {result['score']:.1f}")
        else:
            logging.critical(f"FActScore = {100 * result['score']:.1f}%")
            if "init_score" in result:
                logging.critical(f"FActScore (no length penalty) = {100 * result['init_score']:.1f}%")
        logging.critical(f"Response ratio = {100 * result['respond_ratio']:.1f}%")
        logging.critical(f"Atomic facts per valid response = {result['num_facts_per_response']:.1f}")
        logging.critical(f"Method: {'NLI' if result['using_nli'] else 'LLM'}")

if __name__ == '__main__':
    main()
