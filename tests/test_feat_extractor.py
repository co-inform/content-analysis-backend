#!/usr/bin/env python

"""Tests for `feature extractor` package."""
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, '../src'))
from src import feat_extractor


@pytest.fixture
def sample_text():
    return 'France: !!!!!! ?????? ,,, 10 people dead after shooting at HQ of satirical weekly newspaper #CharlieHebdo, according to witnesses http:\/\/t.co\/FkYxGmuS58'


def test_feature_extraction(sample_text):
    count_feats = feat_extractor.ContentFeats(sample_text).compute()
    assert 6 == count_feats['num_exclamations']
    assert 4 == count_feats['num_commas']
    assert 1 == count_feats['num_dots']
    assert 6 == count_feats['num_exclamations']
