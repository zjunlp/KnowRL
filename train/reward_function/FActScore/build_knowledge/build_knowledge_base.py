import json
import time
import os
import sqlite3
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
import humanize
import platform
import re
from pathlib import Path

# Special separator for document paragraphs
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
# Maximum token length per paragraph
MAX_LENGTH = 256

def split_into_sentences(text):
    """Split text into sentence list, first by paragraphs (\\n\\n), then by English punctuation"""
    if isinstance(text, list):
        sentences = []
        for item in text:
            sentences.extend(split_into_sentences(item))
        return [sent for sent in sentences if sent and str(sent).strip()]
    
    if not isinstance(text, str):
        return [str(text)]
    
    # Handle common abbreviations to avoid treating them as sentence endings
    common_abbr = [
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'St.', 
        'e.g.', 'i.e.', 'etc.', 'vs.', 'U.S.', 'U.K.', 
        'B.C.', 'A.D.', 'Ph.D.', 'M.D.', 'p.m.', 'a.m.'
    ]
    
    # Temporarily replace dots in abbreviations
    for abbr in common_abbr:
        text = text.replace(abbr, abbr.replace('.', '[DOT]'))
    
    # Split by paragraphs first
    paragraphs = re.split(r'\n\n+', text)
    all_sentences = []
    
    # Split sentences within each paragraph
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        # English sentence splitting regex
        pattern = r'(?<=[.!?])\s+'
        
        paragraph_sentences = re.split(pattern, paragraph)
        
        for sentence in paragraph_sentences:
            if not sentence.strip():
                continue
                
            # Restore dots in abbreviations
            for abbr in common_abbr:
                sentence = sentence.replace(abbr.replace('.', '[DOT]'), abbr)
                
            all_sentences.append(sentence.strip())
    
    return all_sentences

def print_section(title):
    """Print formatted section title"""
    line = "=" * 80
    print(f"\n{line}")
    print(f" {title} ".center(80, "="))
    print(f"{line}\n")

def get_db_info(db_path):
    """Get detailed database information"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT SUM(length(title)) + SUM(length(text)) FROM {table_name};")
            size_result = cursor.fetchone()
            size = size_result[0] if size_result[0] is not None else 0
            
            table_info[table_name] = {
                "columns": columns,
                "count": count,
                "size": size
            }
        
        conn.close()
        return table_info
    except Exception as e:
        print(f"Failed to get database info: {str(e)}")
        return {}

def process_document(title, text, titles, output_lines, tokenizer, token_stats, model_max_length):
    """
    Process single document with intelligent sentence-based paragraph splitting
    
    Rules:
    1. Split by paragraphs (\\n\\n) first, then by sentences
    2. Each paragraph should be close to but not exceed MAX_LENGTH tokens
    3. Build paragraphs with complete sentences
    """
    titles.add(title)
    
    sentences = split_into_sentences(text)
    
    passages = []
    current_passage = []
    current_token_count = 0
    total_tokens = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        try:
            sentence_tokens = tokenizer.encode(
                sentence, 
                add_special_tokens=False,
                truncation=True,
                max_length=model_max_length
            )
            sentence_token_count = len(sentence_tokens)
            total_tokens += sentence_token_count
        except:
            try:
                truncated_sentence = sentence[:200] + "..." if len(sentence) > 200 else sentence
                sentence_tokens = tokenizer.encode(
                    truncated_sentence,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=model_max_length
                )
                sentence_token_count = len(sentence_tokens)
                total_tokens += sentence_token_count
                sentence = truncated_sentence
            except:
                continue
        
        # Check if adding this sentence would exceed MAX_LENGTH
        if current_passage and (current_token_count + sentence_token_count > MAX_LENGTH):
            passages.append(" ".join(current_passage))
            current_passage = []
            current_token_count = 0
        
        # Handle oversized sentences
        if sentence_token_count > MAX_LENGTH:
            if current_passage:
                passages.append(" ".join(current_passage))
                current_passage = []
                current_token_count = 0
            
            # Split oversized sentence
            start_idx = 0
            while start_idx < len(sentence_tokens):
                end_idx = min(start_idx + MAX_LENGTH, len(sentence_tokens))
                segment = tokenizer.decode(sentence_tokens[start_idx:end_idx], skip_special_tokens=True)
                passages.append(segment)
                start_idx = end_idx
        else:
            current_passage.append(sentence)
            current_token_count += sentence_token_count
    
    # Save the last paragraph if exists
    if current_passage:
        passages.append(" ".join(current_passage))
    
    # Update token statistics
    token_stats["max_count"] = max(token_stats["max_count"], total_tokens)
    token_stats["total"] += total_tokens
    
    # Join paragraphs with special separator
    processed_text = SPECIAL_SEPARATOR.join(passages)
    
    output_lines.append((title, processed_text))

def build_knowledge_base(db_path, data_path, batch_size=10000, max_docs=None, model_max_length=450):
    """
    Build SQLite knowledge base from JSON or JSONL data file
    
    Args:
        db_path: SQLite database file path
        data_path: Source data file path (JSON or JSONL format)
        batch_size: Batch processing size, default 10000 records
        max_docs: Maximum documents to process, default process all
        model_max_length: Model maximum token length limit, default 450
    """
    print_section("Starting Knowledge Base Construction")
    print(f"Data source: {data_path}")
    print(f"Target database: {db_path}")
    print(f"Batch size: {batch_size} records")
    print(f"Model max token length: {model_max_length}")
    if max_docs:
        print(f"Will only process first {max_docs} documents")
    
    print(f"System: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    file_size = os.path.getsize(data_path)
    print(f"Source file size: {humanize.naturalsize(file_size)}")
    
    # Determine if file is JSON or JSONL format
    is_json_format = False
    with open(data_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1).strip()
        if first_char == '[':
            is_json_format = True
    
    if is_json_format:
        print(f"Detected standard JSON format file")
    else:
        print(f"Detected JSONL format file")
    
    # Connect to SQLite database
    connection = sqlite3.connect(db_path)
    c = connection.cursor()
    
    print_section("Creating Database Schema")
    c.execute("DROP TABLE IF EXISTS documents;")
    c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")
    print("Created table: documents (title PRIMARY KEY, text)")
    
    print_section("Initializing Tokenizer")
    print("Loading RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    titles = set()
    output_lines = []
    tot = 0
    skipped = 0
    failed = 0
    empty_docs = 0
    duplicate_titles = 0
    
    token_stats = {
        "max_count": 0, 
        "total": 0
    }
    
    start_time = time.time()
    
    print_section("Processing Documents")
    
    if is_json_format:
        print("Loading JSON file...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_array = json.load(f)
            
            if not isinstance(data_array, list):
                print("Warning: JSON file doesn't contain array, converting to single object array")
                data_array = [data_array]
            
            total_items = len(data_array)
            print(f"JSON file contains {total_items} entries")
            
            if max_docs and max_docs < total_items:
                data_array = data_array[:max_docs]
                print(f"Will only process first {max_docs} entries")
            
            for i, dp in enumerate(tqdm(data_array, desc="Processing", unit="docs")):
                try:
                    if "title" not in dp or "text" not in dp:
                        skipped += 1
                        continue
                        
                    title = dp["title"]
                    text = dp["text"]
                    
                    if not text or (isinstance(text, str) and not text.strip()):
                        empty_docs += 1
                        skipped += 1
                        continue
                    
                    if title in titles:
                        duplicate_titles += 1
                        skipped += 1
                        continue
                    
                    process_document(title, text, titles, output_lines, tokenizer, token_stats, model_max_length)
                    tot += 1
                    
                    if len(output_lines) >= batch_size:
                        c.executemany("INSERT OR REPLACE INTO documents VALUES (?,?)", output_lines)
                        output_lines = []
                        connection.commit()
                        
                        elapsed_time = time.time() - start_time
                        avg_speed = tot / elapsed_time if elapsed_time > 0 else 0
                        
                        tqdm.write(f"Processed: {tot:,} | Skipped: {skipped:,} | Speed: {avg_speed:.1f} docs/sec")
                        
                except Exception as e:
                    failed += 1
                    if failed < 10:
                        tqdm.write(f"Error processing entry {i+1}: {str(e)}")
                    continue
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {str(e)}")
            return 0
    
    else:
        print("Counting lines...")
        total_lines = sum(1 for _ in open(data_path, 'r', encoding='utf-8'))
        print(f"JSONL file contains {total_lines} lines")
        
        if max_docs:
            print(f"Will only process first {max_docs} lines")
        
        with open(data_path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing", unit="docs")):
                if max_docs and line_num >= max_docs:
                    print(f"Reached maximum processing lines {max_docs}")
                    break
                    
                try:
                    dp = json.loads(line)
                    
                    if "title" not in dp or "text" not in dp:
                        skipped += 1
                        continue
                        
                    title = dp["title"]
                    text = dp["text"]
                    
                    if not text or (isinstance(text, str) and not text.strip()):
                        empty_docs += 1
                        skipped += 1
                        continue
                    
                    if title in titles:
                        duplicate_titles += 1
                        skipped += 1
                        continue
                    
                    process_document(title, text, titles, output_lines, tokenizer, token_stats, model_max_length)
                    tot += 1
                    
                    if len(output_lines) >= batch_size:
                        c.executemany("INSERT OR REPLACE INTO documents VALUES (?,?)", output_lines)
                        output_lines = []
                        connection.commit()
                        
                        elapsed_time = time.time() - start_time
                        avg_speed = tot / elapsed_time if elapsed_time > 0 else 0
                        
                        tqdm.write(f"Processed: {tot:,} | Skipped: {skipped:,} | Speed: {avg_speed:.1f} docs/sec")
                
                except json.JSONDecodeError:
                    failed += 1
                    continue
                except Exception as e:
                    failed += 1
                    if failed < 10:
                        tqdm.write(f"Error processing line {line_num+1}: {str(e)}")
                    continue
    
    # Save remaining records
    if output_lines:
        c.executemany("INSERT OR REPLACE INTO documents VALUES (?,?)", output_lines)
    
    print_section("Creating Indexes")
    print("Creating indexes...")
    c.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);")
    print("Created index: idx_documents_title")
    
    connection.commit()
    connection.close()
    
    elapsed_time = time.time() - start_time
    
    print_section("Knowledge Base Construction Complete")
    
    print(f"Total processing time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")
    print(f"Successful entries: {tot:,}")
    print(f"Skipped entries: {skipped:,}")
    print(f"  - Missing fields: {skipped - empty_docs - duplicate_titles:,}")
    print(f"  - Empty documents: {empty_docs:,}")
    print(f"  - Duplicate titles: {duplicate_titles:,}")
    print(f"Failed entries: {failed:,}")
    print(f"Processing speed: {tot/elapsed_time:.2f} docs/sec")
    
    if tot > 0:
        avg_tokens = token_stats["total"] / tot
        print(f"Max tokens: {token_stats['max_count']:,}")
        print(f"Average tokens: {avg_tokens:.2f}")
    
    db_size = os.path.getsize(db_path)
    print(f"Database file size: {humanize.naturalsize(db_size)}")
    
    print_section("Database Details")
    db_info = get_db_info(db_path)
    for table_name, info in db_info.items():
        print(f"Table: {table_name}")
        print(f"Records: {info['count']:,}")
        print(f"Estimated data size: {humanize.naturalsize(info['size'])}")
        print("Column structure:")
        for col in info['columns']:
            print(f"  - {col[1]} ({col[2]}){' PRIMARY KEY' if col[5] else ''}")
        print()
    
    print(f"Knowledge base construction complete! File saved at: {db_path}")
    return tot

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Base Builder")
    parser.add_argument("--db_path", type=str, 
                      default="./knowledge_base.db", 
                      help="SQLite database file path")
    parser.add_argument("--data_path", type=str, 
                      required=True,
                      help="Source data file path (JSON or JSONL format)")
    parser.add_argument("--batch_size", type=int, 
                      default=10000, 
                      help="Batch processing size (default: 10000)")
    parser.add_argument("--max_docs", type=int,
                      default=None,
                      help="Maximum documents to process (default: process all)")
    parser.add_argument("--max_token_length", type=int,
                      default=450,
                      help="Model maximum token length limit (default: 450)")
    
    args = parser.parse_args()
    
    start = time.time()
    total_docs = build_knowledge_base(
        args.db_path, 
        args.data_path, 
        args.batch_size, 
        args.max_docs, 
        args.max_token_length
    )
    end = time.time()
    print(f"Successfully built knowledge base with {total_docs:,} documents in {end-start:.2f}s")

