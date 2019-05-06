from setuptools import setup

setup(name='troll2vec',
      version='1.1',
      description='First version of TrollBlock project model',
      author='TrollBlock',
      license='MIT',
      packages=[
            'troll2vec',
            'troll2vec/preprocess',
            'troll2vec/typos',
            'toll2vec/tests'
      ],
      install_requires=[
            'pandas==0.23.1',
            'tensorflow==1.12.0',
            'fuzzywuzzy==0.16.0',
            'nltk==3.2.5',
            'gensim==3.4.0',
            'python-Levenshtein==0.12.0',
            'Keras==2.2.0',
            'pymystem3==0.2.0',
            'tqdm==4.23.4',
            'sklearn',
            'distance',
      ],
      zip_safe=False)