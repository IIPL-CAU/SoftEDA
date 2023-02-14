"""
This code was mainly borrowed from https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py
EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
Jason Wei and Kai Zou, EMNLP 2019, https://aclanthology.org/D19-1670.pdf
"""

# Standard Library Modules
import re
import random
import argparse
# 3rd-party Modules
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
# Custom Modules
from utils.utils import set_random_seed

# List of stopwords
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', '']

def get_only_chars(line: str) -> str:
    """
    This function is used to remove all the special characters from the text.

    Args:
        line (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") # Replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) # Remove extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def synonym_replacement(words: list, n: int) -> list:
    """
    Replace n words in the sentence with synonyms from wordnet.

    Args:
        words (list): The list of words in the sentence.
        n (int): The number of words to be replaced.

    Returns:
        list: The list of words in the sentence after replacement.
    """

    new_words = words.copy()

    random_word_list = list(set([word for word in words if word not in stop_words])) # Exclude stop words from being replaced
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word) # Get the synonyms of the words which are not stop words
        if len(synonyms) >= 1: # If there are no synonyms, then skip the word
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: # Only replace up to n words
            break

    # This is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word: str) -> list:
    """
    This is a sub-function of synonym replacement to get synonyms of given word.

    Args:
        word (str): The word to be replaced.

    Returns:
        list: The list of synonyms of the given word.
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm']) # Remove special characters
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word) # Remove the original word from the set of synonyms

    return list(synonyms)

def random_deletion(words: list, p: float) -> list:
    """
    Randomly delete words from the sentence with probability p.

    Args:
        words (list): The list of words in the sentence.
        p (float): The probability of deleting a word.

    Returns:
        list: The list of words in the sentence after deletion.
    """

    # Obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # Randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1) # Generate a random number between 0 and 1
        if r > p: # If the random number is greater than p, then keep the word
            new_words.append(word)
        else:
            #print("deleted", word) # If the random number is less than p, then delete the word
            continue

    # If you end up deleting all words, just return a random word from the original sentence
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

def random_swap(words: list , n: int) -> list:
    """
    Randomly swap two words in the sentence n times.

    Args:
        words (list): The list of words in the sentence.
        n (int): The number of times to swap two words.

    Returns:
        list: The list of words in the sentence after swapping.
    """

    new_words = words.copy()

    for _ in range(n): # Swap the words n times
        new_words = swap_word(new_words)

    return new_words

def swap_word(new_words: list) -> list:
    """
    This is a sub-function of random swap to swap two words in the sentence.

    Args:
        new_words (list): The list of words in the sentence.

    Returns:
        list: The list of words in the sentence after swapping.
    """

    random_idx_1 = random.randint(0, len(new_words)-1) # Get a random index
    random_idx_2 = random_idx_1 # Get another random index
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1) # Make sure the two random indices are different
        counter += 1 # If the two random indices are the same, then try again
        if counter > 3: # If you try more than 3 times, then just return the original sentence
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] # Swap the words

    return new_words

def random_insertion(words: list, n: int) -> list:
    """
    Randomly insert n words into the sentence.

    Args:
        words (list): The list of words in the sentence.
        n (int): The number of words to be inserted.

    Returns:
        list: The list of words in the sentence after insertion.
    """

    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    return new_words

def add_word(new_words: list) -> None:
    """
    This is a sub-function of random insertion to insert a word into the sentence.

    Args:
        new_words (list): The list of words in the sentence.
    """

    synonyms = []
    counter = 0

    while len(synonyms) < 1: # If there are no synonyms, then try again with a different word
        random_word = new_words[random.randint(0, len(new_words)-1)] # Get a random word from the sentence
        synonyms = get_synonyms(random_word) # Get the synonyms of the random word

        counter += 1
        if counter >= 10:
            return

    random_synonym = synonyms[0] # Pick a random synonym from the list of synonyms

    random_idx = random.randint(0, len(new_words)-1) # Get a random index
    new_words.insert(random_idx, random_synonym) # Insert random synonym of a word at the random index

def aeda_random_insertion(words: list, n: int) -> list:
    """
    Following AEDA: An Easier Data Augmentation Technique for Text Classification
    Karimi et al., EMNLP Findings 2021, https://aclanthology.org/2021.findings-emnlp.234.pdf

    This function is also a random inserstion, but only inserts punctuation marks rather than synonyms.

    Args:
        words (list): The list of words in the sentence.
        n (int): The number of punctuation marks to be inserted.
    Return:
        list: The list of words in the sentence after insertion.
    """

    punct_marks = ['.', ';', '?', ':', '!', ','] # AEDA uses only these six punctuation marks
    new_words = words.copy()

    for _ in range(n):
        random_idx = random.randint(0, len(new_words)-1) # Pick a random index to insert the punctuation mark
        random_punct = punct_marks[random.randint(0, len(punct_marks)-1)] # Pick a random punctuation mark to insert

        new_words.insert(random_idx, random_punct) # Insert the punctuation mark at the random index

    return new_words

def run_eda(sentence: str, args: argparse.Namespace) -> str:
    """
    Main function to perform EDA.
    Default value of alpha is 0.1 - Perturb 10% of words in the sentence

    Args:
        sentence (str): The sentence to be augmented.
        args (argparse.Namespace): The arguments passed to the program.

    Returns:
        str: The augmented sentence.
    """
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    len_words = len(words)

    # Augment the sentence with equal probability of each of the 4 operations
    prob = random.uniform(0, 1)

    # Apply the 4 operations with the given probabilities
    if prob < 0.25: # Synonym replacement
        n_sr = max(1, int(args.augmentation_eda_alpha_sr * len_words))
        new_words = synonym_replacement(words, n_sr)
    elif prob < 0.5: # Random swap
        n_rs = max(1, int(args.augmentation_eda_alpha_rs * len_words))
        new_words = random_swap(words, n_rs)
    elif prob < 0.75: # Random insertion
        n_ri = max(1, int(args.augmentation_eda_alpha_ri * len_words))
        new_words = random_insertion(words, n_ri)
    else: # Random deletion
        n_rd = max(1, int(args.augmentation_eda_p_rd * len_words))
        new_words = random_deletion(words, n_rd)

    # Join the words to form the sentence again
    augmented_sentence = ' '.join(new_words)
    return augmented_sentence

def run_aeda(sentence: str, args: argparse.Namespace) -> str:
    """
    Main function to perform AEDA.
    Default value of alpha is 0.1 - Add 10% punctuation marks to the sentence

    Args:
        sentence (str): The sentence to be augmented.
        args (argparse.Namespace): The arguments passed to the program.

    Returns:
        str: The augmented sentence.
    """

    words = sentence.split(' ')
    words = [word for word in words if word != '']
    len_words = len(words)
    n_aeda = max(1, int(args.augmentation_aeda_alpha * len_words))

    new_words = aeda_random_insertion(words, n_aeda)

    # Join the words to form the sentence again
    augmented_sentence = ' '.join(words)
    return augmented_sentence

"""
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1

    #sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    #ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    #rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    #rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    #append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences
"""
