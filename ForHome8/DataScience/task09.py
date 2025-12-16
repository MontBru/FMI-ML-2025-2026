import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import NMF

def main():
    df = pd.read_csv('scrobbler-small-sample.csv')
    artists = pd.read_csv('artists.csv', names=['artist'])

    artists = artists.reset_index().rename(columns={"index": "artist_offset"})
    df = df.merge(artists, on="artist_offset", how="left")

    rows = df["artist_offset"].values     # row indices
    cols = df["user_offset"].values       # column indices
    vals = df["playcount"].values         # data values

    num_artists = df["artist_offset"].max() + 1
    num_users   = df["user_offset"].max() + 1

    A = csr_matrix((vals, (rows, cols)), shape=(num_artists, num_users))

    scaler = MaxAbsScaler()
    A = scaler.fit_transform(A)

    nmf = NMF(n_components=5)
    transformed = nmf.fit_transform(A)

    pos = artists.index[artists["artist"] == "Bruce Springsteen"][0]
    nmf = transformed[pos]
    similarities = transformed.dot(nmf)
    print(artists)

    idx = similarities.argsort()[-10:]

    out_df = pd.DataFrame({
        "artist": artists["artist"].iloc[idx].values,
        "similarity": similarities[idx]
    })

    out_df.to_excel("task09.xlsx", index=False)


        

if __name__ == '__main__':
    main()