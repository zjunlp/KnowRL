import pickle
import os
import time

class LM(object):
    def __init__(self, cache_file=None, read_only_cache=None):
        self.cache_file = cache_file
        self.read_only_cache = read_only_cache
        self.cache_dict = self.load_cache() if cache_file else {}
        self.model = None
        self.add_n = 0

    def load_model(self):
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=4096, max_output_length=4096):
        prompt = prompt.strip()
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()

        generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        if self.cache_file:
            self.cache_dict[cache_key] = generated
            self.add_n += 1
            
        return generated

    def _generate(self, prompt, max_sequence_length=4096, max_output_length=4096):
        raise NotImplementedError()

    def generate_batch(self, prompt_batch, sample_idx=0, max_sequence_length=4096, max_output_length=4096, chunk_size=8):
        prompt_batch = [prompt.strip() for prompt in prompt_batch]
        cache_key_batch = [f"{prompt}_{sample_idx}" for prompt in prompt_batch]
        
        generated_batch = {}
        inference_batch = []
        inference_inds = []
        
        for i, cache_key in enumerate(cache_key_batch):
            if cache_key in self.cache_dict:
                generated_batch[i] = self.cache_dict[cache_key]
            else:
                inference_batch.append(prompt_batch[i])
                inference_inds.append(i)
        
        print(f"{len(generated_batch)} cached, running inference for {len(inference_batch)} items.")
        if len(inference_batch) == 0:
            return [generated_batch[i] for i in range(len(prompt_batch))]
        
        model_load_start = time.time()
        if self.model is None:
            self.load_model()
        model_load_time = time.time() - model_load_start
        print(f"Model load time: {model_load_time:.4f}s")

        generated_texts_list = []
        generated_arrs_list = []
        
        for i in range(0, len(inference_batch), chunk_size):
            batch = inference_batch[i:i+chunk_size]
          
            generated = self._generate_batch(batch, max_sequence_length=max_sequence_length, max_output_length=max_output_length)
                
            if generated:
                generated_texts, generated_arrs = generated
                generated_texts_list.extend(generated_texts)
                generated_arrs_list.extend(generated_arrs)

        for i, ind in enumerate(inference_inds):
            if i < len(generated_texts_list):
                result = (generated_texts_list[i], generated_arrs_list[i])
                generated_batch[ind] = result
                
                if self.cache_file:
                    self.cache_dict[cache_key_batch[ind]] = result

        generated_batch_list = [generated_batch[i] for i in range(len(prompt_batch)) if i in generated_batch]
        
        if self.cache_file:
            self.add_n += len(inference_batch)
            
        return generated_batch_list

    def _generate_batch(self, prompt_batch, max_sequence_length=4096, max_output_length=4096):
        generated_texts = []
        generated_arrs = []
        
        for prompt in prompt_batch:
            text, arr = self._generate(prompt, max_sequence_length, max_output_length)
            generated_texts.append(text)
            generated_arrs.append(arr)
            
        return generated_texts, generated_arrs

    def save_cache(self):
        if self.add_n == 0 or not self.cache_file:
            return

        try:
            if os.path.exists(self.cache_file):
                for k, v in self.load_cache().items():
                    self.cache_dict[k] = v

            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_dict, f)
            print(f"(LM) Saved {self.add_n} cache items to {self.cache_file}")
            self.add_n = 0
        except Exception as e:
            print(f"(LM) Error saving cache: {e}")

    def load_cache(self, allow_retry=True):
        if not self.cache_file:
            return {}
            
        start_time = time.time()
        print(f"(LM) Loading cache from {self.cache_file}")
        
        if not os.path.exists(self.cache_file):
            print(f"(LM) Cache file does not exist, creating new cache")
            return {}
            
        while True:
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                break
            except Exception as e:
                if not allow_retry:
                    print(f"(LM) Failed to load cache: {e}")
                    return {}
                print("Pickle error: retrying in 5 seconds...")
                time.sleep(5)

        print(f"(LM) Loaded {len(cache)} cache items in {time.time() - start_time:.3f} seconds.")
        return cache