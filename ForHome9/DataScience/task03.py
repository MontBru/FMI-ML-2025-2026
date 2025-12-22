import nltk

tweets = ['This is the best #nlp exercise ive found online! #python', '#NLP is super fun! <3 #learning', 'Thanks @datacamp :) #nlp #python']

def main():
    tags_first = nltk.tokenize.regexp_tokenize(tweets[0],r"#\w+")
    print(tags_first)

    mentions_last = nltk.tokenize.regexp_tokenize(tweets[-1], r"@\w+")
    tags_last = nltk.tokenize.regexp_tokenize(tweets[-1], r"#\w+")

    print(f"{mentions_last=}")
    print(f"{tags_last=}")

    tokenizer = nltk.TweetTokenizer()
    all_tokens = tokenizer.tokenize_sents(tweets)

    print(f"{all_tokens=}")


if __name__ == '__main__':
    main()