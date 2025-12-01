from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def main():
    documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
    vectorizer = TfidfVectorizer(input='content')
    X = vectorizer.fit_transform(documents)
    df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    df.to_excel("task06.xlsx", index=True)


if __name__ == '__main__':
    main()