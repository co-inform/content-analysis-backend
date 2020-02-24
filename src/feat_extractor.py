import os
from collections import Counter

import numpy as np
import readability
from moralstrength import string_moral_values
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from spellchecker import SpellChecker
from textblob import TextBlob

from src.preprocessing import normalize_post, handle_twitter_specific_tags

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


class Lexicons:
    def __init__(self, folder='./data/lexicons'):
        # =========== [Mejova et al, 2014] =======================
        with open(os.path.join(folder, "bias-lexicon.txt")) as lex:
            self.bias = set([l.strip() for l in lex])
        # =========== Epistemological bias by [Recasens et al, 2013] ====================
        with open(os.path.join(folder, "assertives.txt")) as lex:
            self.assertives = set([l.strip() for l in lex])
        # entailments
        with open(os.path.join(folder, "implicatives.txt")) as lex:
            self.implicatives = set([l.strip() for l in lex])
        with open(os.path.join(folder, "factives.txt")) as lex:
            self.factives = set([l.strip() for l in lex])
        with open(os.path.join(folder, "hedges.txt")) as lex:
            self.hedges = set([l.strip() for l in lex])
        with open(os.path.join(folder, "report_verbs.txt")) as lex:
            self.report_verbs = set([l.strip() for l in lex])


class TextFeatures:
    def __init__(self, text, tokenizer, lang='en'):
        self.text = text
        self.tokens = tokenizer.tokenize(normalize_post(text))
        self.lang = lang

    def get_structural_feats(self):
        text = self.text
        tokens = self.tokens

        structural_feats = {}
        structural_feats['num_char'] = len(text)
        structural_feats['num_exclamations'] = sum(1 for char in text if char == '!')
        structural_feats['num_commas'] = sum(1 for char in text if char == ',')
        structural_feats['num_dots'] = sum(1 for char in text if char == '.')
        structural_feats['num_questions'] = sum(1 for char in text if char == '?')

        # split text into tokens, default tokenizer is TweetTokenizer
        structural_feats['num_tokens'] = len(tokens)

        # extract pos tags with Penn Treebank Tags
        tags = pos_tag(tokens)
        tag_hist = Counter([tag_hist for token, tag_hist in tags])
        structural_feats['num_NN'] = 0
        structural_feats['num_JJ'] = 0
        structural_feats['num_RB'] = 0
        structural_feats['num_DT'] = 0
        structural_feats['num_CD'] = 0
        structural_feats['num_FW'] = 0
        structural_feats['num_EX'] = 0
        structural_feats['num_MD'] = 0
        structural_feats['num_PR'] = 0
        structural_feats['num_VBPast'] = 0
        structural_feats['num_VBP'] = 0
        structural_feats['num_VBZ'] = 0
        structural_feats['num_VB'] = 0

        for tag, count in tag_hist.items():
            if tag == 'NN' or tag == 'NNP' or tag == 'NN' or tag == 'NNPS' or tag == 'NNS':
                structural_feats['num_NN'] += count
            if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
                structural_feats['num_JJ'] += count
            if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                structural_feats['num_RB'] += count
            if tag == 'DT':
                structural_feats['num_DT'] += count
            if tag == 'CD':
                structural_feats['num_CD'] += count
            if tag == 'FW':
                structural_feats['num_FW'] += count
            if tag == 'EX':
                structural_feats['num_EX'] += count
            if tag == 'MD':
                structural_feats['num_MD'] += count
            if tag == 'PRP' or tag == 'PRP$':
                structural_feats['num_PR'] += count
            if tag == 'VBD' or tag == 'VBN':
                structural_feats['num_VBPast'] += count
            if tag == 'VBP':
                structural_feats['num_VBP'] += count
            if tag == 'VBZ':
                structural_feats['num_VBZ'] += count
            if tag == 'VBG':
                structural_feats['num_VBG'] += count
            if tag == 'VB':
                structural_feats['num_VB'] += count

        structural_feats['num_VBPresent'] = structural_feats['num_VBP'] + structural_feats['num_VBG'] + \
                                            structural_feats['num_VBZ']
        structural_feats['num_VB'] = structural_feats['num_VB'] + structural_feats['num_VBPast'] + structural_feats[
            'num_VBPresent']

    def _count_misspells(self, tokens):
        return len(spell_checker.unknown(tokens))

    def get_sentiment_feats(self):
        return sent_analyzer.polarity_scores(self.text)

    def get_bias_feats(self):
        '''
        credit to https://github.com/BenjaminDHorne/The-NELA-Toolkit
        :return:
        :rtype:
        '''
        bias = {}
        sentences = TextBlob(self.text).sentences
        subjectivity = 0
        for sentence in sentences:
            subjectivity += sentence.sentiment.subjectivity

        bias['subjectivity'] = subjectivity

        tokens = self.tokens
        bigrams = [" ".join(bg) for bg in ngrams(tokens, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(tokens, 3)]
        bias['bias_count'] = float(sum([tokens.count(b) for b in self.bias])) / len(tokens)
        bias['assertives_count'] = float(sum([tokens.count(a) for a in self.assertives])) / len(tokens)
        bias['factives_count'] = float(sum([tokens.count(f) for f in self.factives])) / len(tokens)
        bias['hedges_count'] = sum([tokens.count(h) for h in self.hedges]) + sum(
            [bigrams.count(h) for h in self.hedges]) + sum(
            [trigrams.count(h) for h in self.hedges])
        bias['hedges_count'] = float(bias['hedges_count']) / len(tokens)
        bias['implicatives_count'] = float(sum([tokens.count(i) for i in self.implicatives])) / len(tokens)
        bias['report_verbs_count'] = float(sum([tokens.count(r) for r in self.report_verbs])) / len(tokens)

        return bias

    def get_complexity_feats(self):
        complexity_feats = {}
        # assign text entropy as text complexity
        word_hist = Counter([token for token in self.tokens])
        entropy_sum = 0
        for word, count in word_hist.items():
            entropy_sum += (count * (np.math.log10(self.num_tokens) - np.math.log10(count)))

        complexity_feats['text_complexity'] = (1 / len(self.tokens)) * entropy_sum

        # assign count of misspelling words
        complexity_feats['num_spelling_errors'] = self._count_misspells(self.tokens)

        # assign readability metrics
        readability_metrics = readability.getmeasures(self.text, lang=self.lang)
        complexity_feats['kincaid'] = readability_metrics['readability grades']['Kincaid']
        complexity_feats['ari'] = readability_metrics['readability grades']['ARI']
        complexity_feats['coleman_liau'] = readability_metrics['readability grades']['Coleman-Liau']
        complexity_feats['lix'] = readability_metrics['readability grades']['LIX']
        complexity_feats['flesch'] = readability_metrics['readability grades']['FleschReadingEase']
        complexity_feats['rix'] = readability_metrics['readability grades']['RIX']
        complexity_feats['smog'] = readability_metrics['readability grades']['SMOGIndex']

        return complexity_feats

    def get_social_media_specific_feats(self, social_media='Twitter'):
        social_media_specific_feats = {}
        text = handle_twitter_specific_tags(self.text)
        social_media_specific_feats['num_url'] = text.count('$URL$')

        if social_media == 'Twitter':
            social_media_specific_feats['num_mention'] = text.count('$MENTION$')
            social_media_specific_feats['num_hashtag'] = text.count('$HASHTAG$')

        return social_media_specific_feats

    def get_moral_foundation_feats(self):
        return string_moral_values(self.text)


    def get_emotion_feats(self):
        pass


if __name__ == '__main__':
    text = 'France: :-) :D @twitter frnace !!!!!! ?????? ,,, 10 people dead after shooting at HQ of satirical weekly newspaper #CharlieHebdo, according to witnesses http:\/\/t.co\/FkYxGmuS58'
    tokenizer = TweetTokenizer()
    results = TextFeatures(text, tokenizer).get_moral_foundation_feats()
    print(results)
