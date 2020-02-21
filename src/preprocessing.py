from contractions import contractions_dict
from nltk.stem import SnowballStemmer
import emoji

stemmer = SnowballStemmer('english')


def remove_space(text):
    text = text.strip()
    text = text.split()
    return " ".join(text)


def tokenize(text):
    pass


def correct_spell(text):
    pass


def map_contraction(text):
    for word in contractions_dict.keys():

        if " " + word + " " in text:
            text = text.replace(" " + word + " ", " " + contractions_dict[word] + " ")
    return text

def stem(token):
    return stemmer.stem(token)

def handle_emoji():
    # emojis = emoji.UNICODE_EMOJI
    pass
