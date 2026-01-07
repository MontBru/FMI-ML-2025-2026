from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class WaterDataset(Dataset):
    def __init__(self, csv_filename):
        super().__init__()
        self.df = pd.read_csv(csv_filename)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = np.array(self.df.iloc[idx], dtype=np.float32)
        return (row[:-1], row[-1])
    

def main():
    train_data = WaterDataset("water_train.csv")
    print(f"Number of instances: {len(train_data)}")
    print(f"Fifth item: {train_data[4]}")

    loader = DataLoader(train_data, batch_size=2, shuffle=True)
    first_batch = next(iter(loader))
    print(first_batch)

if __name__ == "__main__":
    main()
