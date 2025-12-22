import pandas as pd
from nltk.corpus import stopwords
import string
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize, regexp_tokenize
from gensim.models import TfidfModel
from collections import defaultdict
import statistics

def bow_to_words(bow, number, dictionary):
    words = bow[:number]
    words = list(map(lambda x : x[0], words))
    words = list(map(lambda x : dictionary[x],words))
    return words

def combine_bows(bows):
    combined = defaultdict(int)
    for bow in bows:
        for token_id, count in bow:
            combined[token_id] += count
    return list(combined.items())

def main():
    df = pd.read_csv('fake_or_real_news.csv')
    df['label'] = df['label'].map(lambda x : 1 if x=='FAKE' else 0)

    percentage_of_fake = df['label'].mean()
    print(percentage_of_fake)


    corpus = df['text']

    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    punct.add("''")
    punct.add("``")
    punct.add("||")


    tokenized = [word_tokenize(doc.lower()) for doc in corpus]
    
    dictionary = Dictionary(tokenized)
    bag_of_words = [dictionary.doc2bow(doc) for doc in tokenized]
    number_of_words_per_doc = [len(doc) for doc in tokenized]


    fake_idx = df.index[df['label'] == 1].tolist()
    real_idx = df.index[df['label'] == 0].tolist()

    fake_bow = combine_bows([bag_of_words[i] for i in fake_idx])
    real_bow = combine_bows([bag_of_words[i] for i in real_idx])

    fake_bow.sort(key=lambda x : x[1], reverse=True)
    real_bow.sort(key=lambda x : x[1], reverse=True)

    top10fake_words = bow_to_words(fake_bow, 10, dictionary)
    top10real_words = bow_to_words(real_bow, 10, dictionary)

    print(top10fake_words)
    print(top10real_words)

    fake_number_of_words = [number_of_words_per_doc[i] for i in fake_idx]
    real_number_of_words = [number_of_words_per_doc[i] for i in real_idx]
    
    print(statistics.mean(fake_number_of_words))
    print(statistics.median(fake_number_of_words))
    print(statistics.mean(real_number_of_words))
    print(statistics.median(real_number_of_words))

    tokenized = [[t for t in doc 
                   if t not in stop_words and t not in punct] 
                    for doc in tokenized]
    
    dictionary = Dictionary(tokenized)
    bag_of_words = [dictionary.doc2bow(doc) for doc in tokenized]
    
    fake_idx = df.index[df['label'] == 1].tolist()
    real_idx = df.index[df['label'] == 0].tolist()

    fake_bow = combine_bows([bag_of_words[i] for i in fake_idx])
    real_bow = combine_bows([bag_of_words[i] for i in real_idx])

    fake_bow.sort(key=lambda x : x[1], reverse=True)
    real_bow.sort(key=lambda x : x[1], reverse=True)

    top10fake_words = bow_to_words(fake_bow, 10, dictionary)
    top10real_words = bow_to_words(real_bow, 10, dictionary)

    print(top10fake_words)
    print(top10real_words)
    

    summary_df = pd.DataFrame({
        "metric": [
            "percentage_fake",
            "mean_words_fake",
            "median_words_fake",
            "mean_words_real",
            "median_words_real"
        ],
        "value": [
            percentage_of_fake,
            statistics.mean(fake_number_of_words),
            statistics.median(fake_number_of_words),
            statistics.mean(real_number_of_words),
            statistics.median(real_number_of_words),
        ]
    })

    # ---- top words BEFORE stopword removal ----
    top_words_raw = pd.DataFrame({
        "fake_top_words_raw": top10fake_words,
        "real_top_words_raw": top10real_words
    })

    # ---- top words AFTER stopword removal ----
    top_words_clean = pd.DataFrame({
        "fake_top_words_clean": top10fake_words,
        "real_top_words_clean": top10real_words
    })

    with pd.ExcelWriter("data_audit.xlsx", engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        top_words_raw.to_excel(writer, sheet_name="top_words_raw", index=False)
        top_words_clean.to_excel(writer, sheet_name="top_words_clean", index=False)



    # tfidf = TfidfModel(bag_of_words)
    # tfidf_corpus = tfidf[bag_of_words]

    # scores = defaultdict(float)

    # for doc in tfidf_corpus:
    #     for token_id, weight in doc:
    #         scores[token_id] += weight

    # print("first 5 ids:")
    # print(f"{[(dictionary[i], scores[i]) for i in range(5)]}")

    # top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # top5_words = [(dictionary[id], score) for id, score in top5]
    # print(f"{top5_words=}")


if __name__ == '__main__':
    main()