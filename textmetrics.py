import os
import re

from bs4 import BeautifulSoup

from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords, preprocess_string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from textstat.textstat import textstatistics #, legacy_round
import math


transform_to_lower = lambda s: s.lower()

remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)


# Filters to be executed in pipeline
CLEAN_FILTERS = [strip_tags,
                 strip_punctuation,
                 strip_multiple_whitespaces,
                 transform_to_lower,
                 remove_stopwords,
                 remove_single_char]


# Method does the filtering of all the unrelevant text elements
def cleaning_pipe(document):
    # Invoking gensim.parsing.preprocess_string method with set of filters
    processed_words = preprocess_string(document, CLEAN_FILTERS)

    return processed_words


def joinList(processed_words):
    return ' '.join(processed_words)


# Splits the text into sentences, using
# Spacy's sentence segmentation which can
# be found at https://spacy.io/usage/spacy-101

def load_nlp(version = 'en_core_web_sm'):
    nlp = spacy.load(version)
    return nlp

def break_sentences_optimized(text,nlp):
    doc = nlp(text)
    return list(doc.sents)

def break_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return list(doc.sents)


# Returns Number of Words in the text
def word_count(sentences_list):
    #sentences = break_sentences(text)
    words = 0
    for sentence in sentences_list:
        words += len([token for token in sentence])
    return words


# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)


# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    if sentences == 0:
        return 0
    average_sentence_length = float(words / sentences)
    return average_sentence_length
def avg_sentence_length_optimized(words,sentences):
    if sentences == 0:
        return 0
    average_sentence_length = float(words / sentences)
    return average_sentence_length


# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return textstatistics().syllable_count(word)


# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)

    if words == 0:
        return None

    ASPW = float(syllable) / float(words)
    return math.floor(ASPW)


def avg_syllables_per_word_optimized(count_words,count_syllables):
    if count_words == 0:
        return None

    ASPW = float(count_syllables) / float(count_words)
    return math.floor(ASPW)


# Return total Difficult Words in a text
def difficult_words(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]

    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)


def difficult_words_optimized(sentences,nlp):
    words = []
    for sentence in sentences:
        words += [str(token) for token in sentence]

    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)


# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words
# present in the text
def poly_syllable_count(text):
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]

    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count


def flesch_reading_ease(text):
    """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
        ASL = average sentence length (number of words
                divided by number of sentences)
        ASW = average word length in syllables (number of syllables
                divided by number of words)
    """
    avg_sent_len = avg_sentence_length(text)
    if avg_sent_len == 0:
        return None

    avg_sya_word = avg_syllables_per_word(text)
    if avg_sya_word == 0:
        return None

    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - \
          float(84.6 * avg_syllables_per_word(text))
    return math.floor(FRE)

def flesch_reading_ease_optimized(avg_sent_len,avg_sya_word):
    """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
        ASL = average sentence length (number of words
                divided by number of sentences)
        ASW = average word length in syllables (number of syllables
                divided by number of words)
    """

    if avg_sent_len == 0:
        return None

    if avg_sya_word == 0:
        return None

    FRE = 206.835 - float(1.015 * avg_sent_len) - \
          float(84.6 * avg_sya_word)
    return math.floor(FRE)


def gunning_fog(text):
    count_word = word_count(text)

    if count_word == 0:
        return None
    per_diff_words = (difficult_words(text) / count_word * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade

def gunning_fog_optimized(count_words,count_difficult_words,average_sentence_length):
    if count_words == 0:
        return None
    per_diff_words = (count_difficult_words / count_words * 100) + 5
    grade = 0.4 * (average_sentence_length + per_diff_words)
    return grade


def syllables_counter(word):
    ts = textstatistics()
    return ts.syllable_count(word)


def smog_index(text):
    """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here,
        polysyllable count = number of words of more
        than two syllables in a sample of 30 sentences.
    """

    if sentence_count(text) >= 3:
        poly_syllab = syllables_counter(text)
        SMOG = (1.043 * (30 * (poly_syllab / sentence_count(text))) ** 0.5) \
               + 3.1291
        return math.floor(SMOG, 1)
    else:
        return 0

def smog_index_optimized(count_sentences,count_syallables):
    """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here,
        polysyllable count = number of words of more
        than two syllables in a sample of 30 sentences.
    """

    if count_sentences >= 3:
        SMOG = (1.043 * (30 * (count_syallables / count_sentences)) ** 0.5) \
               + 3.1291
        return math.floor(SMOG)
    else:
        return 0


def dale_chall_readability_score(text):
    """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
    """
    words = word_count(text)

    if words == 0:
        return None
    # Number of words not termed as difficult words
    diffwords = difficult_words(text)

    count = words - diffwords
    if words > 0:
        # Percentage of words not on difficult word list

        per = float(count) / float(words) * 100

    # diff_words stores percentage of difficult words
    diff_words = 100 - per

    raw_score = (0.1579 * diff_words) + \
                (0.0496 * avg_sentence_length(text))

    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score

    if diff_words > 5:
        raw_score += 3.6365

    return math.floor(raw_score)

def dale_chall_readability_score_optimized(count_words,count_difficult_words,average_sentence_length):
    """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
    """

    if count_words == 0:
        return None
    # Number of words not termed as difficult words

    count = count_words - count_difficult_words
    if count_words > 0:
        # Percentage of words not on difficult word list

        per = float(count) / float(count_words) * 100

    diff_words = 100 - per

    raw_score = (0.1579 * diff_words) + \
                (0.0496 * average_sentence_length)

    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score

    if diff_words > 5:
        raw_score += 3.6365

    return math.floor(raw_score)


def process_metrics(text):

    text_metrics = {
        'textblock_id': [],
        "blob_sentiment_polarity": [],
        "blob_sentiment_subjectivity": [],
        "vader_sia_neg": [],
        "vader_sia_neu": [],
        "vader_sia_pos": [],
        "vader_sia_compound": [],
        "score_fog_gunning": [],
        "score_flesh": [],
        "score_smog": [],
        "score_dalle": []
    }

    nlp = load_nlp('en_core_web_sm')
    nlp.max_length = len(text) + 100


    blob_sentiment_polarity = 0
    blob_sentiment_subjectivity = 0
    vader_sia = 0
    vader_sia_neg = 0
    vader_sia_neu = 0
    vader_sia_pos = 0
    vader_sia_compound = 0
    count_difficult_words = 0
    score_fog_gunning = 0
    count_syllables = 0
    average_syllable_word = 0
    score_flesh = 0
    score_smog = 0
    score_dalle = 0

    # TEXT BLOB
    blob = TextBlob(text)
    blob_sentiment_polarity = blob.sentiment.polarity
    blob_sentiment_subjectivity = blob.sentiment.subjectivity

    # VADER
    sid_obj = SentimentIntensityAnalyzer()
    vader_sia = sid_obj.polarity_scores(text)
    vader_sia_neg = vader_sia['neg']
    vader_sia_neu = vader_sia['neu']
    vader_sia_pos = vader_sia['pos']
    vader_sia_compound = vader_sia['compound']


    # TEXT METRICS OPTIMIZED
    break_sentences_list = break_sentences_optimized(text,nlp)
    count_sentences = len(break_sentences_list)
    count_words = word_count(break_sentences_list)
    count_syllables = syllables_count(text)
    average_syllable_word = avg_syllables_per_word_optimized(count_words,count_syllables)
    average_sentence_length = avg_sentence_length_optimized(count_words,count_sentences)
    count_difficult_words = difficult_words_optimized(break_sentences_list,nlp)

    score_fog_gunning = gunning_fog_optimized(count_words,count_difficult_words,average_sentence_length)
    score_flesh = flesch_reading_ease_optimized(average_sentence_length,average_syllable_word)
    score_smog = smog_index_optimized(count_sentences,count_syllables)
    score_dalle = dale_chall_readability_score_optimized(count_words,count_difficult_words,average_sentence_length)
    
    # TEXT METRICS
    text_metrics["blob_sentiment_polarity"].append(blob_sentiment_polarity)
    text_metrics["blob_sentiment_subjectivity"].append(blob_sentiment_subjectivity)
    text_metrics["vader_sia_neg"].append(vader_sia_neg)
    text_metrics["vader_sia_neu"].append(vader_sia_neu)
    text_metrics["vader_sia_pos"].append(vader_sia_pos)
    text_metrics["vader_sia_compound"].append(vader_sia_compound)
    text_metrics["score_fog_gunning"].append(score_fog_gunning)
    text_metrics["score_flesh"].append(score_flesh)
    text_metrics["score_smog"].append(score_smog)
    text_metrics["score_dalle"].append(score_dalle)
    
    return text_metrics

def load_extract_text(filepath):
    raw_file = None
    with open(filepath,encoding="utf-8") as file:
        raw_file = file.read()

    text = BeautifulSoup(raw_file, 'lxml').text
    text = text.replace('\n', '')
    return text, raw_file