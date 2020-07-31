import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

# The sentence came from the New York Times
ex = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile pohone market and ordered the company to alte its practices"

# we apply word tokenization and part-of-speech tagging
def preprocess(sentence):
	sent = nltk.word_tokenize(sentence)
	sent = nltk.pos_tag(sentence)
	return sent

sent = preprocess(ex)
# print(sent)

# implement noun phrase chunking to identify named intities using a regular expression
pattern = "NP: {<DT>?<JJ>*<NN>}"

# create a chunk parser
cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
# print(cs)

# representing chunk structures in files
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)
