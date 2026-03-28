import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Typo injection: randomly replace a vowel in ~30% of words with an adjacent
    # QWERTY key. This is a "reasonable" transformation because real users make
    # typos, but it disrupts BERT's tokenizer by producing out-of-vocabulary
    # subword pieces — causing a larger accuracy drop than synonym replacement.
    qwerty_neighbors = {
        'a': 'qwsz', 'e': 'wrds', 'i': 'uojk', 'o': 'iplk', 'u': 'yhji',
        'A': 'QWSZ', 'E': 'WRDS', 'I': 'UOJK', 'O': 'IPLK', 'U': 'YHJI',
    }
    vowels = set('aeiouAEIOU')

    words = word_tokenize(example["text"])
    new_words = []
    for word in words:
        if random.random() < 0.30 and any(c in vowels for c in word):
            # Pick a random vowel position in the word and replace it
            vowel_indices = [i for i, c in enumerate(word) if c in vowels]
            idx = random.choice(vowel_indices)
            char = word[idx]
            replacement = random.choice(qwerty_neighbors.get(char, char))
            word = word[:idx] + replacement + word[idx+1:]
        new_words.append(word)
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
