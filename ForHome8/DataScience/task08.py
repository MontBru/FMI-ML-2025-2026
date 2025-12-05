import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
import os

def save_digit_plot(digit, filename):
    """Save a single digit (13x8) plot to a PNG."""
    digit = np.array(digit).reshape(13, 8)

    plt.figure(figsize=(2, 3))  # small image
    plt.imshow(digit, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    df = pd.read_csv('lcd-digits.csv')

    # Create Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Plots"

    row = 1  # start inserting images here

    # --- Plot the first digit ---
    first_filename = "first_digit.png"
    save_digit_plot(df.loc[0], first_filename)
    ws.add_image(XLImage(first_filename), f"A{row}")
    row += 20  # spacing between images

    #I would do 7 components for NMF, because these 
    #are images of 7-segment displays.

    # --- NMF components ---
    nmf = NMF(n_components=7)
    nmf.fit_transform(df)

    for i, component in enumerate(nmf.components_):
        fname = f"nmf_component_{i}.png"
        save_digit_plot(component, fname)
        ws.add_image(XLImage(fname), f"A{row}")
        row += 20

    # --- PCA components ---
    pca = PCA(n_components=7)
    pca.fit_transform(df)

    for i, component in enumerate(pca.components_):
        fname = f"pca_component_{i}.png"
        save_digit_plot(component, fname)
        ws.add_image(XLImage(fname), f"A{row}")
        row += 20

    # Save Excel file
    wb.save("task08.xlsx")

    # Optional: clean up temporary PNGs
    for file in os.listdir():
        if file.endswith(".png"):
            os.remove(file)


if __name__ == '__main__':
    main()
