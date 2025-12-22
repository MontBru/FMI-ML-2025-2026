import re
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk import pos_tag
import matplotlib.pyplot as plt

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')


from nltk.corpus import stopwords
import string

from collections import Counter
import spacy


def filter_tokens(tokenized):
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    punct.add("''")
    punct.add("``")
    punct.add("||")

    tokenized = [
        t for t in tokenized
        if t not in stop_words and t not in punct
    ]

def show_pie_chart(entity_counts):
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        entity_counts.values(),
        autopct='%1.1f%%',
        startangle=90
    )

    plt.legend(
        wedges,
        entity_counts.keys(),
        title="Entity Type",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    plt.axis("equal")
    plt.show()

def main():
    with open("article_uber.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentences = re.findall(r"[^.!?]*[.!?]", text)
    tokenized = [word_tokenize(sentence) for sentence in sentences]
    tagged = nltk.pos_tag_sents(tokenized)
    ne_chunked = nltk.ne_chunk_sents(tagged, binary=False)
    flat_chunked = [x for sub in ne_chunked for x in sub]

    entity_counts = Counter()
    for chunk in flat_chunked:
        if isinstance(chunk, nltk.Tree):
            entity_counts[chunk.label()] += 1

    show_pie_chart(entity_counts)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entity_counts = Counter(ent.label_ for ent in doc.ents)

    show_pie_chart(entity_counts)

    #entity types in nltk: GPE, Person, Organization
    #entity types in spacy: Organization, Person, Loc, NORP, Money

    #Answer: C


if __name__ == '__main__':
    main()