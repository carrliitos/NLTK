import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
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
# pprint(iob_tagged)

# recognize named entities using a classifier and convert tagged sequences into a chunk tree
ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
# print(ne_tree)

# we use SpaCy - trained on the OntoNotes 5 corpus
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp("European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices")
# printing on an entity level
# pprint([(X.text, X.label_) for X in doc.ents])

# printing on token-level entity annotation using BILUO tagging scheme
# pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
'''B means the token begins an entity, I means it is inside an enitity, O means it is outside an entity, and an empty one means no entity tag is set.'''

# We use BeautifulSoup4 for extractions from an article
from bs4 import BeautifulSoup
import requests
import re

def urlToString(url):
	res = requests.get(url)
	html = res.text
	soup = BeautifulSoup(html, "html5lib")
	for script in soup(["script", "style", "aside"]):
		script.extract()
	return " ".join(re.split(r'[\n\t]+', soup.get_text()))

ny_bb = urlToString("https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news")
article = nlp(ny_bb)
len(article.ents) # there are currently 158 entities in the article
# represent each entity as 10 unique labels
labels = [x.label_ for x in article.ents]
Counter(labels)

# find the three most frequent tokens
items = [x.text for x in article.ents]
Counter(items).most_common(3)

# Randomly select one sentence and generate the raw markup
sentences = [x for x in article.sents]
# print(sentences[21]) # "A spokeswoman for the F.B.I. did not respond to a message seeking comment about why Mr. Strzok was dismissed rather than demoted."

# These following will not work uless we're using Jupyter Notebook or an IDE that supports SpaCy's builtin displaCy visualizer
# displacy.render(nlp(str(sentences[21])), jupyter=True, style="ent")
# displacy.render(nlp(str(sentencens[21])), jupyter=True, style="dep")
# options = ["distance": 120]

# extract POS and lemmatize the selected sentence
verbatim = [(x.orth_, x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[21])) if not y.is_stop and y.pos_ != "PUNCT"]]
# print(verbatim)

# Additional options: we can visualize the whole thing if we wanted. Since we're working on a text editor, let's not.
