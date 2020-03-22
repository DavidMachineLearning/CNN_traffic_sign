from tensorflow.python.keras import backend as K


def f1_score(y_true, y_pred):
    """ F1 score metric """

    def recall_metric(labels, predictions):
        """ Recall metric """
        true_positives = K.sum(K.round(K.clip(labels * predictions, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(labels, 0, 1)))
        recall_score = true_positives / (possible_positives + K.epsilon())
        return recall_score

    def precision_metric(labels, predictions):
        """ Precision metric """
        true_positives = K.sum(K.round(K.clip(labels * predictions, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(predictions, 0, 1)))
        precision_score = true_positives / (predicted_positives + K.epsilon())
        return precision_score

    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)

    # K.epsilon() is a small number used to prevent division by zero
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
