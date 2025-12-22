import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize, regexp_tokenize

from nltk.corpus import stopwords
import string

def main():
    with open("article.txt", "r", encoding="utf-8") as f:
        text = f.read()

    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    punct.add("''")
    punct.add("``")
    tokenized = word_tokenize(text.lower())
    tokenized = [
        t for t in tokenized
        if t not in stop_words and t not in punct
    ]
    
    dictionary = Dictionary([tokenized])
    bag_of_words = dictionary.doc2bow(tokenized)
    bag_of_words.sort(key=lambda x : x[1], reverse=True)

    top10words = bag_of_words[:10]
    top10words = list(map(lambda x : x[0], top10words))
    top10words = list(map(lambda x : dictionary[x],top10words))

    print(top10words)

    # The article covers the topic of debugging or something similar that is software related


if __name__ == '__main__':
    main()