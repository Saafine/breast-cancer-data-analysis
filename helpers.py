import matplotlib.pyplot as plt
import seaborn as sn

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
