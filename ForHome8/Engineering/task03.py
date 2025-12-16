import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from ml_lib.feature_extraction.text import TfidfVectorizer
import pandas as pd

def main():
    documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.vocab
    )
    df.to_excel("./ForHome8/Engineering/task03_engineering.xlsx", index=True)


if __name__ == '__main__':
    main()