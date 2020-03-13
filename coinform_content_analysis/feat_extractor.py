import os
from collections import Counter

import numpy as np
from afinn import Afinn
from moralstrength import string_moral_values
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from readability import Readability
from spellchecker import SpellChecker
from textblob import TextBlob
import nltk

from coinform_content_analysis.preprocessing import normalize_post, handle_twitter_specific_tags

'''
This package contains content features for measuring the credibility.
generic content features proposed by Olteanu et al. [2013]
Notes:
1 - in the paper, sentiment analysis is measured with polarity of the page, num of negative, positive, subjective and objective sentences. We use rule based sentiment analyzer (Vader)
in order to compute sentiment of the sentences
2- We add additional readability metrics rather than smog
3- We exclude the informativeness since it is not applicable for tweets
'''

nltk.download('vader_lexicon')

vader_sent_analyzer = SentimentIntensityAnalyzer()
spell_checker = SpellChecker(distance=2)
afinn_sent_analyzer = Afinn(emoticons=True)
tokenizer = TweetTokenizer()


def _count_misspells(tokens):
    return len(spell_checker.unknown(tokens))


class Lexicons:
    def __init__(self, folder):
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


lexicon = Lexicons(folder='data/lexicons')


class TextFeatures(object):
    @staticmethod
    def get_structural_feats(text):
        tokens = tokenizer.tokenize(normalize_post(text))
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
        structural_feats['num_VBG'] = 0
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
        return structural_feats

    @staticmethod
    def get_sentiment_feats(text):
        sentiment = {}
        sentiment['afinn'] = afinn_sent_analyzer.score(text)
        vader_scores = vader_sent_analyzer.polarity_scores(text)
        sentiment['vader_pos'] = vader_scores['pos']
        sentiment['vader_neg'] = vader_scores['neg']
        sentiment['vader_comp'] = vader_scores['compound']
        sentiment['vader_neu'] = vader_scores['neu']
        return sentiment

    @staticmethod
    def get_bias_feats(text):
        '''
        credit to https://github.com/BenjaminDHorne/The-NELA-Toolkit
        :return:
        :rtype:
        '''
        bias = {}
        subjectivity = -100
        if text:
            sentences = TextBlob(text).sentences
            subjectivity = 0
            for sentence in sentences:
                subjectivity += sentence.sentiment.subjectivity

        bias['subjectivity'] = subjectivity

        tokens = tokenizer.tokenize(normalize_post(text))
        bigrams = [" ".join(bg) for bg in ngrams(tokens, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(tokens, 3)]
        num_tokens = len(tokens)
        bias['bias_count'] = float(sum([tokens.count(b) for b in lexicon.bias])) / num_tokens if num_tokens > 0 else 0
        bias['assertives_count'] = float(
            sum([tokens.count(a) for a in lexicon.assertives])) / num_tokens if num_tokens > 0 else 0
        bias['factives_count'] = float(
            sum([tokens.count(f) for f in lexicon.factives])) / num_tokens if num_tokens > 0 else 0
        bias['hedges_count'] = sum([tokens.count(h) for h in lexicon.hedges]) + sum(
            [bigrams.count(h) for h in lexicon.hedges]) + sum(
            [trigrams.count(h) for h in lexicon.hedges])
        bias['hedges_count'] = float(bias['hedges_count']) / len(tokens) if num_tokens > 0 else 0
        bias['implicatives_count'] = float(
            sum([tokens.count(i) for i in lexicon.implicatives])) / num_tokens if num_tokens > 0 else 0
        bias['report_verbs_count'] = float(
            sum([tokens.count(r) for r in lexicon.report_verbs])) / num_tokens if num_tokens > 0 else 0

        return bias

    @staticmethod
    def get_complexity_feats(text, lang='en'):
        complexity_feats = {}
        tokens = tokenizer.tokenize((text))
        num_tokens = len(tokens)
        # assign text entropy as text complexity
        word_hist = Counter([token for token in tokens])
        entropy_sum = 0
        for word, count in word_hist.items():
            entropy_sum += (count * (np.math.log10(num_tokens) - np.math.log10(count)))

        complexity_feats['text_complexity'] = (1 / num_tokens) * entropy_sum if num_tokens > 0 else -100

        # assign count of misspelling words
        complexity_feats['num_spelling_errors'] = _count_misspells(tokens)

        # assign readability metrics
        readability_metrics = Readability(text)
        try:
            complexity_feats['flesch-kincaid'] = readability_metrics.flesch_kincaid().score
        except:
            complexity_feats['flesch-kincaid'] = -100
        try:
            complexity_feats['ari'] = readability_metrics.ari().score
        except:
            complexity_feats['ari'] = -100
        try:
            complexity_feats['coleman_liau'] = readability_metrics.coleman_liau().score
        except:
            complexity_feats['coleman_liau'] = -100
        try:
            complexity_feats['flesch'] = readability_metrics.flesch().score
        except:
            complexity_feats['flesch'] = -100

        try:
            complexity_feats['smog'] = readability_metrics.smog().score
        except:
            complexity_feats['smog'] = -100

        return complexity_feats

    @staticmethod
    def get_social_media_specific_feats(text, social_media='Twitter'):
        social_media_specific_feats = {}
        text = handle_twitter_specific_tags(text)

        if social_media == 'Twitter':
            social_media_specific_feats['num_mention'] = text.count('$MENTION$')
            social_media_specific_feats['num_hashtag'] = text.count('$HASHTAG$')

        return social_media_specific_feats

    @staticmethod
    def get_linked_feat(text):
        text = handle_twitter_specific_tags(text)
        return {'num_url': text.count('$URL$')}

    @staticmethod
    def get_moral_foundation_feats(text):
        return string_moral_values(text)

    @staticmethod
    def get_emotion_feats(text):
        pass

    @staticmethod
    def get_factuality(text):
        pass

    def get_all_feats(self, text):
        feats = {}
        feats['bias'] = self.get_bias_feats(text, lexicon).values()
        feats['moral_foundation'] = self.get_moral_foundation_feats(text).values()
        feats['complexity'] = self.get_complexity_feats(text).values()
        # todo implement this
        # self.get_emotion_feats()
        # self.get_factuality()
        feats['linked_feat'] = self.get_linked_feat(text).values()
        feats['structural'] = self.get_structural_feats(text).values()
        return feats

    def get_all_feat_names(self):
        return ['bias', 'moral_foundation', 'complexity', 'linked_feat', 'structural']


class Engagements:
    pass


class Media:
    pass


if __name__ == '__main__':
    text = 'France: :-) :D @twitter frnace !!!!!! ?????? ,,, 10 people dead after shooting at HQ of satirical weekly newspaper #CharlieHebdo, according to witnesses http:\/\/t.co\/FkYxGmuS58'
    lexicon = Lexicons()
    results = TextFeatures().get_bias_feats(text, lexicon)
    print(results)
