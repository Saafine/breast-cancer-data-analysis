import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from i18n import COLUMN_HEADER_TO_LABEL


def run_data_analysis():
    db_path = "./db/breast-cancer-wisconsin.data"
    missing_values = "?"
    df = pd.read_csv(db_path, na_values=missing_values)
    df.drop(["sample"], axis=1, inplace=True)
    printBasicColumnData(df)
    plotClassDistribution(df)


def printBasicColumnData(df):
    for column in df.columns.values:
        min = df[column].min()
        max = df[column].max()
        mean = round(df[column].mean(), 2)
        missing = df[column].isnull().sum()
        plotValuesDistributionForColumn(column, COLUMN_HEADER_TO_LABEL.get(column))
        print(f'Column: {column}, min: {min}, max: {max}, mean: {mean}, missing: {missing}')


def plotClassDistribution(df):
    label_value_counts = df["class"].value_counts()
    labels = 'Rak łagodny', 'Nowotwór złośliwy'
    sizes = list(label_value_counts)
    explode = (0, 0.1)  # only "explode" the 2nd slice
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def plotValuesDistributionForColumn(df, column='clump-thickness', title="Grubość guza"):
    plt.style.use('seaborn-deep')
    benign = df.loc[df['class'] == 2][[column]].to_numpy().ravel()
    malignant = df.loc[df['class'] == 4][[column]].to_numpy().ravel()
    plt.hist([benign, malignant], label=["Rak łagodny", "Nowotwór Złośliwy"])
    plt.legend()
    plt.title(title)
    plt.xlabel("Wartość")
    plt.ylabel("Ilość")
    plt.show()


def plotConfusionMatrix(confusion_matrix, title):
    labels = ["Rak Łagody", "Nowotwór Złośliwy"]
    sn.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Przewidziana klasyfikacja")
    plt.ylabel("Rzeczywista klasyfikacja")
    plt.title(title)
    plt.show()


def plotClassifiersAccuracy(classifiers_accuracy):
    plt.style.use('ggplot')
    labels = list(map(lambda clf: clf[0], classifiers_accuracy))
    values = list(map(lambda clf: clf[1] * 100, classifiers_accuracy))
    x_pos = [i for i, _ in enumerate(labels)]
    plt.bar(x_pos, values, color='green')
    plt.xlabel("Klasyfikator")
    plt.ylabel("Skuteczność %")
    plt.title("Porównanie skuteczności klasyfikatorów")

    plt.xticks(x_pos, labels)
    plt.ylim([90, 100])

    plt.show()
