import re
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk import pos_tag

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')


from nltk.corpus import stopwords
import string


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

def main():
    with open("article_uber.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentences = re.findall(r"[^.!?]*[.!?]", text)
    last_sentence = sentences[-1]

    tokenized_last = word_tokenize(last_sentence)
    tagged_last = pos_tag(tokenized_last)
    print(tagged_last)


    tokenized_first = word_tokenize(sentences[0])
    tagged_first = pos_tag(tokenized_first)
    
    for tree in nltk.ne_chunk(tagged_first, binary=True):
        print(tree)


    tokenized = [word_tokenize(sentence) for sentence in sentences]
    tagged = nltk.pos_tag_sents(tokenized)
    # flat_tagged = [x for sub in tagged for x in sub]
    
    ne_chunked = nltk.ne_chunk_sents(tagged, binary=True)
    flat_chunked = [x for sub in ne_chunked for x in sub]
    
    print(f"{flat_chunked[:10]=}")

    def f(x):
        return isinstance(x, nltk.Tree)

    named_entities = list(filter(f, flat_chunked))
    named_entities = list(map(lambda x : " ".join(w for w, _ in x.leaves()),named_entities))

    print(f"{named_entities=}")


if __name__ == '__main__':
    main()