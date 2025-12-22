# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')


from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize, regexp_tokenize
from gensim.models import TfidfModel


from nltk.corpus import stopwords
import string
import ast
from collections import defaultdict

def combine_bows(bows):
    combined = defaultdict(int)
    for bow in bows:
        for token_id, count in bow:
            combined[token_id] += count
    return list(combined.items())

def bow_to_words(bow, number, dictionary):
    words = bow[:number]
    words = list(map(lambda x : x[0], words))
    words = list(map(lambda x : dictionary[x],words))
    return words


def main():
    with open("messy_articles.txt", "r", encoding="utf-8") as f:
        content = f.read()

    corpus = ast.literal_eval(content)

    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    punct.add("''")
    punct.add("``")
    punct.add("||")

    tokenized = [[t for t in doc 
                  if t not in stop_words and t not in punct] 
                    for doc in corpus]
    
    dictionary = Dictionary(tokenized)
    bag_of_words = [dictionary.doc2bow(doc) for doc in tokenized]
    
    first10word_ids_fifth_doc = bag_of_words[4][:10]

    for doc in bag_of_words:
        doc.sort(key=lambda x : x[1], reverse=True)

    top5words_fifth_doc = bow_to_words(bag_of_words[4], 5, dictionary)
    # bag_of_words[4][:5]
    # top5words_fifth_doc = list(map(lambda x : x[0], top5words_fifth_doc))
    # top5words_fifth_doc = list(map(lambda x : dictionary[x],top5words_fifth_doc))

    combined_bow = combine_bows(bag_of_words)
    combined_bow.sort(key=lambda x : x[1], reverse=True)

    top5words_all_docs = bow_to_words(combined_bow, 5, dictionary)

    print(dictionary.token2id['computer'])
    print(f"{first10word_ids_fifth_doc=}")
    print(f"{top5words_fifth_doc=}")
    print(f"{top5words_all_docs=}")

    tfidf = TfidfModel(bag_of_words)
    tfidf_corpus = tfidf[bag_of_words]

    scores = defaultdict(float)

    for doc in tfidf_corpus:
        for token_id, weight in doc:
            scores[token_id] += weight

    print("first 5 ids:")
    print(f"{[(dictionary[i], scores[i]) for i in range(5)]}")

    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    top5_words = [(dictionary[id], score) for id, score in top5]
    print(f"{top5_words=}")



if __name__ == '__main__':
    main()