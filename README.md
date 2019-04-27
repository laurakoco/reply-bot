# Reply Bot

Smart reply bot implemented in Python.

## What Does It Do?

The objective of this project is to create a bot that suggests short responses to a text or email message using Natural Language Processing. A reply bot is a type of chatbot. The goal of a chatbot is to mimic written or spoken speech as to simulate a conversation with a human.

The scope of bot is casual conversation.

From a high-level, there are two variants of chatbots, **rule-based** and **self-learning**:
* A **rule-based** bot responds via pattern matching and rules.
* A **self-learning** bot responds via using machine learning.
    * A **retrieval-based** chatbot replies by locating the best response from a database (corpus).
    * A **generative-based** chatbot use deep learning methods to respond and can generate a response it has never seen before.

In order to understand the tradeoffs between the different types of chatbots, I implemented a rule-based, a retrieval-based and a machine learning-based reply bot.

## Built With

* Python
* [NLTK](https://www.nltk.org/)
* TensorFlow and TFLearn

## Design

### Rule-based bot

A rule-based chatbot is the simplest type of bot. This type of bot searches for predefined patterns in the input and uses a set of rules (if-else logic) to suggest replies. The suggested replies, in this case, are predefined. We can implement pattern matching on an input message by using regular expressions and applying rules with if-else statements.

For example, to search for a greeting we can use this as our regex pattern:

```python
greeting_str = 'hi|hello|sup|hey|good\s+[morning|afternoon|evening]
```

If the pattern is found, we suggest predefined greeting responses:

```python
greeting_response = ["hi", "hey", "hello"]
```

Altogether we have

```python
if re.search(greeting_str, user_input):
    return greeting_response
```

For the reply bot, I have defined 8 simple rules:
* Greeting
* Goodbye
* How are you
* Thank you
* Do you/will you/can you/would you
* Are you
* When
* What’s up

Each rule has an associated reply:

```python
greeting_response = ["hi", "hello", "hey"]
goodbye_response = ["bye", "talk to you later"]
thank_response = ['happy to help','don\'t mention it','my pleasure']
inquiry_response = ['i\'m doing ok','ok','i\'ve been better']
future_response = ['yes','no','maybe']
what_you_response = ['nothing', 'not much']
are_you_response = ['yes','no', 'maybe']
when_response = ['soon','not now']
no_response = ['[No Suggestion]']
```

## Usage

Execute the program

```
python rule_based_reply.py
```

The program will then request a message

```
* Hello! Type in a message and I will suggest some replies! If you'd like to exit please type quit!
>>>
```

Try different different messages and the reply bot will generate suggested replies or [No Suggestion]

```
>>> hello
* hi
* hey
* hello
```

To quit, type quit

```
>>> quit
```

## Author

**Laura Kocubinski** [laurakoco](https://github.com/laurakoco)

## Acknowledgments

* Boston University MET Master Science Computer Science Program
* MET CS 664 Artificial Intelligence
