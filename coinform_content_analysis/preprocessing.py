import re

from contractions import contractions_dict
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

URL_REGEX = '(https?\:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S'
MENTION_REGEX = '([@][\w_-]+)'
HASHTAG_REGEX = '([#][\w_-]+)'


def remove_space(text):
    text = text.strip()
    text = text.split()
    return " ".join(text)


def handle_twitter_specific_tags(text):
    text = re.sub(URL_REGEX, '$URL$', text)
    text = re.sub(MENTION_REGEX, '$MENTION$', text)
    text = re.sub(HASHTAG_REGEX, '$HASHTAG$', text)
    return text


def normalize_post(text):
    text = remove_space(text)
    text = handle_twitter_specific_tags(text)
    text = text.replace('$URL$', '')
    text = text.replace('$MENTION$', '')
    text = text.replace('$HASHTAG$', '')
    return text


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
