import matplotlib.pyplot as plt
from utils.utils import count_unique_elements


def plot_distribution(labels, title=""):
    """
    Function used to plot the distribution of the dataset.

    Args:
        labels (numpy.ndarray): the labels of the given dataset (i.e. y_train, must be an array of scalar values)
        title (str): the title showed in the plot
    """
    data_distribution = count_unique_elements(labels)
    plt.figure(figsize=(14, 4))
    plt.bar(tuple(data_distribution.keys()), tuple(data_distribution.values()))
    plt.title(title)
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
