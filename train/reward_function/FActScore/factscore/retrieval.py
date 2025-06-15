import json
import time
import os
import sqlite3
import numpy as np
import pickle as pkl
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import concurrent.futures
import logging
import threading
from functools import lru_cache

class LogConfig:
    VERBOSE = False
    _lock = threading.Lock()
    _initialized = False

    @staticmethod
    def setup_logging():
        with LogConfig._lock:
            if not LogConfig._initialized:
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                LogConfig._initialized = True

    @staticmethod
    def log(message, level="info", force=False):
        LogConfig.setup_logging()

        if not LogConfig.VERBOSE and not force and level != "error":
            return

        with LogConfig._lock:
            if level == "info":
                logging.info(message)
            elif level == "warning":
                logging.warning(message)
            elif level == "error":
                logging.error(message)
            elif level == "debug":
                if LogConfig.VERBOSE:
                    logging.debug(message)

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

class DocDB(object):
    def __init__(self, db_path=None):
        self.db_path = db_path

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file {self.db_path} does not exist")

        LogConfig.log(f"Loading knowledge base: {self.db_path}", force=True)
        start_time = time.time()

        db_size = os.path.getsize(self.db_path)
        with tqdm(total=100, desc="Loading database", ncols=80) as pbar:
            pbar.update(10)
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

            # Database optimization settings
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.execute("PRAGMA cache_size = 10000")

            pbar.update(40)

            # Check if database has tables
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            if len(cursor.fetchall()) == 0:
                self.close()
                raise ValueError(f"Database {self.db_path} is empty, contains no tables")

            pbar.update(40)

            try:
                cursor.execute("SELECT COUNT(*) FROM documents;")
                count = cursor.fetchone()[0]
                db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            except:
                pass

            pbar.update(10)

        elapsed = time.time() - start_time
        LogConfig.log(f"Knowledge base loaded, time: {elapsed:.2f}s", force=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        return self.db_path

    def close(self):
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

    @lru_cache(maxsize=100)
    def get_text_from_title_cached(self, title):
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()

        if not results or len(results) == 0:
            raise ValueError(f"Document with title '{title}' not found in database")

        paragraphs = results[0][0].split(SPECIAL_SEPARATOR)

        if len(paragraphs) == 0:
            raise ValueError(f"Document with title '{title}' has no valid content")

        return paragraphs

    def get_text_from_title(self, title):
        paragraphs = self.get_text_from_title_cached(title)
        results = [{"title": title, "text": para} for para in paragraphs]
        return results


class Retrieval(object):
    def __init__(self, db_path=None, cache_path=None, embed_cache_path=None,
                 retrieval_type="gtr-t5-large", batch_size=None):
        self.db_path = db_path
        self.db = DocDB(db_path) if db_path else None
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size or 32

        if retrieval_type != "bm25" and not retrieval_type.startswith("gtr-"):
            raise ValueError(f"Unsupported retrieval type: {retrieval_type}, must be bm25 or start with gtr-")

        self.encoder = None
        self.cache = {}
        self.embed_cache = {}

        if cache_path or embed_cache_path:
            self.load_cache()

        self.add_n = 0
        self.add_n_embed = 0

        LogConfig.log(f"Retrieval initialized, type: {retrieval_type}", level="info", force=True)

    def load_encoder(self):
        if self.encoder is not None:
            return

        LogConfig.log("Loading encoder...", level="info", force=True)
        from sentence_transformers import SentenceTransformer

        start_time = time.time()
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder

        elapsed = time.time() - start_time
        LogConfig.log(f"Encoder loaded, time: {elapsed:.2f}s", level="info", force=True)

    def load_cache(self):
        def _load_cache_file():
            if self.cache_path and os.path.exists(self.cache_path):
                LogConfig.log(f"Loading retrieval cache from {self.cache_path}...", level="info")
                try:
                    with open(self.cache_path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    LogConfig.log(f"Failed to load retrieval cache: {str(e)}", level="warning", force=True)
            return {}

        def _load_embed_cache_file():
            if self.embed_cache_path and os.path.exists(self.embed_cache_path):
                LogConfig.log(f"Loading embedding cache from {self.embed_cache_path}...", level="info")
                try:
                    with open(self.embed_cache_path, "rb") as f:
                        return pkl.load(f)
                except Exception as e:
                    LogConfig.log(f"Failed to load embedding cache: {str(e)}", level="warning", force=True)
            return {}

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            cache_future = executor.submit(_load_cache_file)
            embed_cache_future = executor.submit(_load_embed_cache_file)

            self.cache = cache_future.result()
            self.embed_cache = embed_cache_future.result()

        elapsed = time.time() - start_time
        LogConfig.log(f"Cache loaded, retrieval: {len(self.cache)} items, embedding: {len(self.embed_cache)} items, time: {elapsed:.2f}s", level="info")

    def save_cache(self):
        if (self.add_n == 0 and self.add_n_embed == 0) or (not self.cache_path and not self.embed_cache_path):
            return

        def _save_cache_file():
            if self.add_n > 0 and self.cache_path:
                LogConfig.log(f"Saving {self.add_n} new retrieval cache items...", level="info")

                existing_cache = {}
                if os.path.exists(self.cache_path):
                    try:
                        with open(self.cache_path, "r") as f:
                            existing_cache = json.load(f)
                    except:
                        pass

                existing_cache.update(self.cache)
                with open(self.cache_path, "w") as f:
                    json.dump(existing_cache, f)

                return True
            return False

        def _save_embed_cache_file():
            if self.add_n_embed > 0 and self.embed_cache_path:
                LogConfig.log(f"Saving {self.add_n_embed} new embedding cache items...", level="info")

                existing_embed_cache = {}
                if os.path.exists(self.embed_cache_path):
                    try:
                        with open(self.embed_cache_path, "rb") as f:
                            existing_embed_cache = pkl.load(f)
                    except:
                        pass

                existing_embed_cache.update(self.embed_cache)
                with open(self.embed_cache_path, "wb") as f:
                    pkl.dump(existing_embed_cache, f)

                return True
            return False

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            cache_future = executor.submit(_save_cache_file)
            embed_cache_future = executor.submit(_save_embed_cache_file)

            cache_saved = cache_future.result()
            embed_saved = embed_cache_future.result()

            if cache_saved:
                self.add_n = 0
            if embed_saved:
                self.add_n_embed = 0

        elapsed = time.time() - start_time
        if cache_saved or embed_saved:
            LogConfig.log(f"Cache saved, time: {elapsed:.2f}s", level="info")

    def get_bm25_passages(self, topic, query, passages, k):
        if topic in self.embed_cache:
            bm25 = self.embed_cache[topic]
        else:
            tokenized_corpus = [psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages]
            bm25 = BM25Okapi(tokenized_corpus)
            self.embed_cache[topic] = bm25
            self.add_n_embed += 1

        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        indices = np.argsort(-scores)[:k]

        return [passages[i] for i in indices]

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()

        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]

            passage_vectors = np.zeros((len(inputs), self.encoder.get_sentence_embedding_dimension()), dtype=np.float32)

            def encode_batch(batch_idx):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(inputs))
                batch_inputs = inputs[start_idx:end_idx]

                batch_vectors = self.encoder.encode(
                    batch_inputs,
                    batch_size=self.batch_size,
                    device=self.encoder.device,
                    show_progress_bar=False
                )

                return (start_idx, end_idx, batch_vectors)

            num_batches = (len(inputs) + self.batch_size - 1) // self.batch_size

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(encode_batch, batch_idx) for batch_idx in range(num_batches)]

                for future in concurrent.futures.as_completed(futures):
                    start_idx, end_idx, batch_vectors = future.result()
                    passage_vectors[start_idx:end_idx] = batch_vectors

            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1

        query_vectors = self.encoder.encode([retrieval_query],
                                          batch_size=self.batch_size,
                                          device=self.encoder.device,
                                          show_progress_bar=False)[0]

        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]

        return [passages[i] for i in indices]

    def get_passages(self, topic, question, k=5):
        if self.db is None:
            raise ValueError("Please set knowledge base path first")

        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key in self.cache:
            LogConfig.log(f"Using cached retrieval result (topic: '{topic}')", level="debug")
            return self.cache[cache_key]

        LogConfig.log(f"Retrieving... topic: '{topic}', question: '{question}'", level="info")

        passages = self.db.get_text_from_title(topic)

        if self.retrieval_type == "bm25":
            results = self.get_bm25_passages(topic, retrieval_query, passages, k)
        else:
            results = self.get_gtr_passages(topic, retrieval_query, passages, k)

        self.cache[cache_key] = results
        self.add_n += 1

        # Periodic cache saving
        if self.add_n % 10 == 0 or self.add_n_embed % 10 == 0:
            threading.Thread(target=self.save_cache).start()

        return results







