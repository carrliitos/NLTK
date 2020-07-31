import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# The sentence came from the New York Times
ex = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile pohone market and ordered the company to alte its practices"

# we apply word tokenization and part-of-speech tagging
def preprocess(sentence):
	sent = nltk.word_tokenize(sentence)
	sent = nltk.pos_tag(sentence)
	return sent

sent = preprocess(ex)
print(sent)
