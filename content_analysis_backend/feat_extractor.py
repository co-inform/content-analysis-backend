from nltk import word_tokenize
'''
This package contains content features for measuring the credibility.
generic credibility features proposed by Olteanu et al. [2013]
    #exclamations
    #commas
    #dots
    #questions
    #token_count
    ?polarity
    #positive
    #negative
    #subjective
    #objective
    #spelling_errors
    @text_complexity
    @informativeness
    @smog
    category
    #NN
    #VB
    #JJ
    #RB
    #DT
'''


class GenericContentFeats:
    def __init__(self, text):
        self.text = text
        self.num_exclamations = None
        self.num_commas = None
        self.num_dots = None
        self.num_questions = None
        self.num_tokens = None
        self.polarity = None
        self.positive = None
        self.negative = None
        self.subjective = None
        self.objective = None
        self.num_spelling_errors = None
        self.text_complexity = None
        self.informativeness = None
        self.smog = None
        self.category = None
        self.num_NN = None
        self.num_VB = None
        self.num_JJ = None
        self.num_RB = None
        self.num_DT = None

    def compute(self):
        self.num_exclamations = count_exclamations(self.text)
        self.num_commas = count_commas(self.text)
        self.num_dots = count_dots(self.text)
        self.num_questions = count_questions(self.text)
        self.num_tokens = count_tokens(self.text)


def count_exclamations(text):
    return sum(1 for char in text if char == '!')


def count_commas(text):
    return sum(1 for char in text if char == ',')


def count_dots(text):
    return sum(1 for char in text if char == '.')


def count_questions(text):
    return sum(1 for char in text if char == '?')


def count_tokens(text):
    return len(word_tokenize(text))


