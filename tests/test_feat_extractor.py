#!/usr/bin/env python

"""Tests for `feature extractor` package."""
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir,'../content_analysis_backend'))
from content_analysis_backend import feat_extractor

@pytest.fixture
def sample_text():
    return 'France: !!!!!! ?????? ,,, 10 people dead after shooting at HQ of satirical weekly newspaper #CharlieHebdo, according to witnesses http:\/\/t.co\/FkYxGmuS58'


def test_feature_extraction(sample_text):
    assert 6 == feat_extractor.count_exclamations(sample_text)
    assert 4 == feat_extractor.count_commas(sample_text)
    assert 1 == feat_extractor.count_dots(sample_text)
    assert 6 == feat_extractor.count_exclamations(sample_text)


