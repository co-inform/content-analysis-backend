========================
content-analysis-backend
========================
1- Install requirements: pip install -r requirements.txt

install the project:
```
bash install.sh
```

extract features:
```
bash feature_extractor.sh
```

run experiments on classical models:
```
bash classical_experiments.sh
```

run experiments on transformers:
```
bash neural_experiments.sh
```

.. image:: https://img.shields.io/pypi/v/content_analysis_backend.svg
        :target: https://pypi.python.org/pypi/content_analysis_backend

.. image:: https://img.shields.io/travis/isspek/content_analysis_backend.svg
        :target: https://travis-ci.com/isspek/content_analysis_backend

.. image:: https://readthedocs.org/projects/content-analysis-backend/badge/?version=latest
        :target: https://content-analysis-backend.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/isspek/content_analysis_backend/shield.svg
     :target: https://pyup.io/repos/github/isspek/content_analysis_backend/
     :alt: Updates



This resposiory contains source code of content analysis module which assigns credibility score of given tweet. 


* Free software: MIT license
* Documentation: https://content-analysis-backend.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Acknowledgments
---------------

Credibility features Olteanu et al [2013]:


    @inproceedings{DBLP:conf/ecir/OlteanuPLA13,
      author    = {Alexandra Olteanu and
                   Stanislav Peshterliev and
                   Xin Liu and
                   Karl Aberer},
      title     = {Web Credibility: Features Exploration and Credibility Prediction},
      booktitle = {{ECIR}},
      series    = {Lecture Notes in Computer Science},
      volume    = {7814},
      pages     = {557--568},
      publisher = {Springer},
      year      = {2013}
    }


Readability metrics:
    https://github.com/andreasvc/readability
    
This work is sponsored by EU Project Horizon 2020 [CoInform](https://coinform.eu/)
