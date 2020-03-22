import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix", classes=tuple(range(43)),
                          cmap=plt.cm.Blues, save_flg=False):
    """
    This function is used to display the confusion metric.

    Args:
        y_true (tf.Tensor or numpy.ndarray): labels
        y_pred (tf.Tensor or numpy.ndarray): predictions from the network
        title (str): title used in the plot
        classes (tuple or list): class indices
        cmap (matplotlib.pyplot.cm): color map used in the plot
        save_flg (bool): if true, the image is saved on disk
    """

    # check that y_true and y_pred are not one hot encoded vector
    if len(y_true.shape) != 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) != 1:
        y_pred = np.argmax(y_pred, axis=1)

    con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=3)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(17, 17))
    sns.heatmap(con_mat_df, annot=True, cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figure.tight_layout()

    if save_flg:
        plt.savefig("confusion_matrix.png")

    plt.show()
