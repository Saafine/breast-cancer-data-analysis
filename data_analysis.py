import pandas as pd
import matplotlib.pyplot as plt
from i18n import COLUMN_HEADER_TO_LABEL

db_path = "./db/breast-cancer-wisconsin.data"
missing_values = "?"
df = pd.read_csv(db_path, na_values=missing_values)
df.drop(["sample"], axis=1, inplace=True)

def plotValuesDistributionForColumn(column='clump-thickness', title="Grubość guza"):
    plt.style.use('seaborn-deep')
    benign = df.loc[df['class'] == 2][[column]].to_numpy().ravel()
    malignant = df.loc[df['class'] == 4][[column]].to_numpy().ravel()
    plt.hist([benign, malignant], label=["Rak łagodny", "Nowotwór Złośliwy"])
    plt.legend()
    plt.title(title)
    plt.xlabel("Wartość")
    plt.ylabel("Ilość")
    plt.show()

def printBasicColumnData():
    for column in df.columns.values:
        min = df[column].min()
        max = df[column].max()
        mean = round(df[column].mean(), 2)
        median = df[column].median()
        missing = df[column].isnull().sum()
        plotValuesDistributionForColumn(column, COLUMN_HEADER_TO_LABEL.get(column))
        print(f'Column: {column}, min: {min}, max: {max}, avg: {mean}, median: {median} missing: {missing}')


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


def run_data_analysis():
    printBasicColumnData()
    plotClassDistribution()

run_data_analysis()
