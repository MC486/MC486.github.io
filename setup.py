from setuptools import setup, find_packages

setup(
    name="word-game",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.8.1",
        "wordfreq>=3.0.3",
        "pyyaml>=6.0.1",
        "pytest>=7.4.3",
        "scikit-learn>=1.4.0",
        "tensorflow>=2.15.0",
    ],
) 