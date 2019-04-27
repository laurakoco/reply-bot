# Reply Bot

Smart reply bot implemented in Python.

## What Does It Do?

The objective of this project is to create a bot that suggests short responses to a text or email message. A reply bot is a type of chatbot. The goal of a chatbot is to mimic written or spoken speech as to simulate a conversation with a human.

This bot is designed to reply to casual conversation. This will require both analyzing and generating text which will be done using Natural Language Processing (NLP).

From a high-level, there are two variants of chatbots, **rule-based** and **self-learning**:
* A rule-based bot responds via pattern matching and rules.
* A self-learning bot responds via using machine learning.
    * A retrieval-based chatbot replies by locating the best response from a database.
    * A generative-based chatbot use deep learning methods to respond and can generate a response it has never seen before.

In this project, I have implemented a rule-based, retrieval-based and machine learning-based reply bot.

## Built With

* Python
* [NLTK](https://www.nltk.org/)
* TensorFlow and TFLearn

## Usage

Execute the program

```python
python rule_based_reply.py
* Hello! Type in a message and I will suggest some replies! If you'd like to exit please type quit!
>>>
```

Try different phrases and the reply bot will generate suggested replies

```python
>>> hello
>>> hello
* hi
* hey
* hello
```

To quit, simple type quit

```python
>>> quit
```

## Author

**Laura Kocubinski** [laurakoco](https://github.com/laurakoco)

## Acknowledgments

* Boston University MET Master Science Computer Science Program
* MET CS 664 Artificial Intelligence
