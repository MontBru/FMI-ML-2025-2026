import pandas as pd

def main():
    df = pd.read_csv('scrobbler-small-sample.csv')
    artists = pd.read_csv('artists.csv', names=['artists'])

    

if __name__ == '__main__':
    main()