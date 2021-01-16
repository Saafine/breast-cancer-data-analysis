import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from i18n import COLUMN_HEADER_TO_LABEL

sns.set_theme()
import numpy as np
np.random.seed(0)

db_path = "./db/breast-cancer-wisconsin.data"
missing_values = "?"
df = pd.read_csv(db_path, na_values=missing_values)
df.drop(["sample"], axis=1, inplace=True)


def printBasicColumnData():
    total_values = df["class"].count()
    for column in df.columns.values:
        min = df[column].min()
        max = df[column].max()
        mean = round(df[column].mean(), 2)
        missing = df[column].isnull().sum()
        plotValuesDistributionForColumn(column, COLUMN_HEADER_TO_LABEL.get(column))
        print(f'Column: {column}, min: {min}, max: {max}, mean: {mean}, missing: {missing}')


def plotClassDistribution():
    label_value_counts = df["class"].value_counts()
    labels = 'Rak łagodny', 'Nowotwór złośliwy'
    sizes = list(label_value_counts)
    explode = (0, 0.1)  # only "explode" the 2nd slice
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def plotValuesDistributionForColumn(column='bare-nuclei', title="Bare Nuclei"):
    benign = df.loc[df['class'] == 2][[column]]
    malignant = df.loc[df['class'] == 4][[column]]

    plt.hist(benign, label="Rak Łagodny", alpha=1)
    plt.hist(malignant, label="Nowotwór Złośliwy", alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel("Wartość")
    plt.ylabel("Ilość")
    plt.show()


# printBasicColumnData()
# plotClassDistribution()
plotValuesDistributionForColumn()
