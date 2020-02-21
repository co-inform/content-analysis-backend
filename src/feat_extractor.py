import re
from collections import Counter

import emoji
import numpy as np
import readability
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer
from spellchecker import SpellChecker
from textblob import TextBlob
from emojipedia import Emojipedia

'''
This package contains content features for measuring the credibility.
generic content features proposed by Olteanu et al. [2013]
Notes:
1 - in the paper, sentiment analysis is measured with polarity of the page, num of negative, positive, subjective and objective sentences. We use rule based sentiment analyzer (Vader)
in order to compute sentiment of the sentences
2- We add additional readability metrics rather than smog
3- We exclude the informativeness since it is not applicable for tweets
'''

sent_analyzer = SentimentIntensityAnalyzer()
spell_checker = SpellChecker(distance=2)

URL_REGEX = '(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S'
MENTION_REGEX = '([@][\w_-]+)'
HASHTAG_REGEX = '([#][\w_-]+)'


class ContentFeats:
    def __init__(self, text, lang='en'):
        self.text = text
        self.lang = lang
        # ====== Generic Content feats proposed by Olteanu et al. [2013] ======
        self.num_exclamations = 0
        self.num_commas = 0
        self.num_dots = 0
        self.num_questions = 0
        self.num_tokens = 0
        self.polarity_pos = 0.0
        self.polarity_neg = 0.0
        self.polarity_neutral = 0.0
        self.polarity_compound = 0.0
        self.subjectivity = 0.0
        self.num_spelling_errors = 0
        self.text_complexity = None
        self.informativeness = None
        # readability measurement
        self.smog = None
        self.category = None
        # NN, NNP, NN, NNPS, NNS
        self.num_NN = 0
        # VB, VBD, VBG, VBN, VBP, VBZ
        self.num_VB = 0
        # JJ, JJR, JJS
        self.num_JJ = 0
        # adverbs RB, RBR(adv comparative), RBS(adv superlative)
        self.num_RB = 0
        # determiner
        self.num_DT = 0
        self.num_char = 0

        # ====== Task specific feats ======
        # numeral, cardinal
        self.num_CD = 0
        # foreign word
        self.num_FW = 0
        # existential
        self.num_EX = 0
        # model auxiliary eg. can, must
        self.num_MD = 0
        # pronoun PRP (e.g hers, herself), PRP$ (e.g her, his, mine)
        self.num_PR = 0
        # past tense (VBD), past participle (VBN)
        self.num_VBPast = 0
        # present tense (VBG), present participle (VBG), present not 3rd person (VBP), present third person (VBZ)
        self.num_VBPresent = 0
        # present not third person
        self.num_VBP = 0
        # present third person
        self.num_VBZ = 0
        # present participant
        self.num_VBG = 0
        # readability measurements
        self.kincaid = None
        self.ari = None
        self.coleman_liau = None
        self.lix = None
        self.flesch = None
        self.rix = None
        self.num_url = 0
        self.num_mention = 0
        self.num_hashtag = 0

    def compute(self, tokenizer=TweetTokenizer()):
        self.num_char = len(self.text)
        # @todo fix this
        self.text = re.sub(URL_REGEX, '$URL$', self.text)
        self.text = re.sub(MENTION_REGEX, '$MENTION$', self.text)
        self.text = re.sub(HASHTAG_REGEX, '$HASHTAG$', self.text)

        self.num_url = self.text.count('$URL$')
        self.num_mention = self.text.count('$MENTION$')
        self.num_hashtag = self.text.count('$HASHTAG$')

        # clean tweet from hashtags, urls, and mentions
        self.text = self.text.replace('$HASHTAG$', '')
        self.text = self.text.replace('$URL$', '')
        self.text = self.text.replace('$MENTION$', '')

        self.num_exclamations = sum(1 for char in self.text if char == '!')
        self.num_commas = sum(1 for char in self.text if char == ',')
        self.num_dots = sum(1 for char in self.text if char == '.')
        self.num_questions = sum(1 for char in self.text if char == '?')

        # split text into tokens, default tokenizer is TweetTokenizer
        tokens = tokenizer.tokenize(self.text)
        self.num_tokens = len(tokens)

        # extract pos tags with Penn Treebank Tags
        tags = pos_tag(tokens)
        tag_hist = Counter([tag_hist for token, tag_hist in tags])
        for tag, count in tag_hist.items():
            if tag == 'NN' or tag == 'NNP' or tag == 'NN' or tag == 'NNPS' or tag == 'NNS':
                self.num_NN += count
            if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                self.num_JJ += count
            if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                self.num_RB += count
            if tag == 'DT':
                self.num_DT += count
            if tag == 'CD':
                self.num_CD += count
            if tag == 'FW':
                self.num_FW += count
            if tag == 'EX':
                self.num_EX += count
            if tag == 'MD':
                self.num_MD += count
            if tag == 'PRP' or tag == 'PRP$':
                self.num_PR += count
            if tag == 'VBD' or tag == 'VBN':
                self.num_VBPast += count
            if tag == 'VBP':
                self.num_VBP += count
            if tag == 'VBZ':
                self.num_VBZ += count
            if tag == 'VBG':
                self.num_VBP += count
            if tag == 'VB':
                self.num_VB += count

        self.num_VBPresent = self.num_VBP + self.num_VBG + self.num_VBZ
        self.num_VB = self.num_VB + self.num_VBPast + self.num_VBPresent

        # sentiment analysis
        sent_results = sent_analyzer.polarity_scores(self.text)
        self.polarity_pos = sent_results['pos']
        self.polarity_neg = sent_results['neg']
        self.polarity_neutral = sent_results['neu']
        self.polarity_compound = sent_results['compound']

        # assign count of misspelling words
        self.num_spelling_errors = self._count_misspells(tokens)

        # assign readability metrics
        readability_metrics = readability.getmeasures(self.text, lang=self.lang)
        self.kincaid = readability_metrics['readability grades']['Kincaid']
        self.ari = readability_metrics['readability grades']['ARI']
        self.coleman_liau = readability_metrics['readability grades']['Coleman-Liau']
        self.lix = readability_metrics['readability grades']['LIX']
        self.flesch = readability_metrics['readability grades']['FleschReadingEase']
        self.rix = readability_metrics['readability grades']['RIX']
        self.smog = readability_metrics['readability grades']['SMOGIndex']

        # assign text entropy as text complexity
        word_hist = Counter([token for token in tokens])
        entropy_sum = 0
        for word, count in word_hist.items():
            entropy_sum += (count * (np.math.log10(self.num_tokens) - np.math.log10(count)))

        self.text_complexity = (1 / len(tokens)) * entropy_sum

        # assign subjectivity
        sentences = TextBlob(self.text).sentences
        subjectivity = 0
        for sentence in sentences:
            subjectivity += sentence.sentiment.subjectivity
        self.subjectivity = subjectivity / len(sentences)

        # assign text category
        # @TODO

        # convert emojis to text
        self.text = emoji.demojize(self.text)
        self.text = self.text.replace("::", " ")  # for emojis that don't have space between them

        return vars(self)

    def _count_misspells(self, tokens):
        return len(spell_checker.unknown(tokens))


def compute_user_cred(user):
    pass


def computer_domain_cred(domain):
    pass


if __name__ == '__main__':
    text = 'France: :-) :D @twitter frnace !!!!!! ?????? ,,, 10 people dead after shooting at HQ of satirical weekly newspaper #CharlieHebdo, according to witnesses http:\/\/t.co\/FkYxGmuS58'
    results = ContentFeats(text).compute()
    print(results)
    print(len(results))
