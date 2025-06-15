import torch
import re
import numpy as np
import string
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


MONTHS = ["january", "february", "march", "april", "may", "june", "july", 
          "august", "september", "october", "november", "december"]

def detect_initials(text):
    """Detect initials in text (e.g., A.B.)"""
    pattern = r"[A-Z]\. ?[A-Z]\."
    return [m for m in re.findall(pattern, text)]

def fix_sentence_splitter(curr_sentences, initials=None):
    """Fix sentence splitting issues, especially handling initials and short sentences"""
    # If no initials provided, detect initials in all sentences
    if initials is None:
        initials = []
        for sent in curr_sentences:
            initials.extend(detect_initials(sent))
            
    # Handle incorrect splitting caused by initials
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # Merge sentences i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    
    sentences = []
    combine_with_previous = None
    
    # Handle short sentences and sentences not starting with uppercase
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            sentences.append(sent)
            
    return sentences






