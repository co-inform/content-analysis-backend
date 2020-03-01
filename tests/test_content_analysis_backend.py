#!/usr/bin/env python

"""Tests for `coinform_content_analysis` package."""
import pytest
import os
import json
from pathlib import Path


def read_content(tweet_path):
    data = None
    with open(tweet_path) as f:
        data = json.load(f)['text']
    return data

@pytest.fixture
def conversation():
    sample_dir = Path(os.path.dirname(__file__))  / 'sample'
    conversation_structure_path = list(Path(sample_dir).glob('*/structure.json'))[0]
    with open(conversation_structure_path) as f:
        flatten_conversation = {}
        conversation_structure = json.load(f)
        source_fname = conversation_structure_path.parent.name + '.json'
        source_path = conversation_structure_path.parent / 'source-tweet' / source_fname
        flatten_conversation['source'] = read_content(source_path)

        # take only first depth reply, we don't use reply to reply tweets
        replies = conversation_structure[conversation_structure_path.parent.name]
        flatten_conversation['replies'] = []
        for reply_id, reply in replies.items():
            reply_fname = reply_id + '.json'
            reply_path = conversation_structure_path.parent / 'replies' / reply_fname
            if reply_path.exists():
                flatten_conversation['replies'].append(read_content(reply_path))
    return flatten_conversation


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_feature_extraction(conversation):
    assert len(conversation['replies']) > 0



