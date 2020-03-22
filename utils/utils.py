from PIL import Image
import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt


def count_unique_elements(arr):
    """
    Function used to count the number of samples for each unique element in the array

    Args:
         arr (np.ndarray): an array of scalar values

    Returns:
        (dict): a dictionary of unique values (keys) and number of occurrences (values)
    """
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))


def preprocess_image(img, convert_to_gray=False):
    """
    Function used for the pre-processing step of the input images.
    You can choose between returning an rgb image or a gray scale one.

    If returning a rgb image:
        applies histogram equalization on each channel and then stack the channels back together.

    If returning a gray scale image:
        the image is first converted to grayscale, then Histogram Equalization is applied.

    The image is also normalized by dividing it by 255.

    Args:
        img(np.ndarray): an rgb image with pixels in range 0-255
        convert_to_gray (bool): If true, the image is converted to grayscale and return a gray scale image

    Returns:
        (np.ndarray): the preprocessed image
    """
    if convert_to_gray:
        # Convert to grayscale and normalize
        preprocessed_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply Histogram Equalization
        preprocessed_image = np.reshape(cv2.equalizeHist(preprocessed_image) / 255, (32, 32, 1))

    else:
        r = np.reshape(cv2.equalizeHist(img[:, :, 0]) / 255, (32, 32, 1))
        g = np.reshape(cv2.equalizeHist(img[:, :, 1]) / 255, (32, 32, 1))
        b = np.reshape(cv2.equalizeHist(img[:, :, 2]) / 255, (32, 32, 1))
        preprocessed_image = np.concatenate([r, g, b], axis=2)

    return preprocessed_image


def label_smoothing(labels, factor=0.12):
    """
    This function applies label smoothing.

    Args:
        labels (np.ndarray): one hot encoded labels
        factor (float): smoothing factor

    Returns:
        labels (np.ndarray): smoothed labels
    """

    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


def predict_from_web(url, data, model1, model2, model3, crop=None):
    """
    Function used to make predictions of images taken from the web.

    Args:
         url (str): url link where the image is located
         data (pandas.DataFrame): data frame containing labels names
         model1 (tf.keras.Model): model used for the prediction
         model2 (tf.keras.Model): model used for the prediction
         model3 (tf.keras.Model): model used for the prediction
         crop (tuple): (y-min, y-max, x-min, x-max)
    """

    r = requests.get(url, stream=True)
    img_raw = np.asarray(Image.open(r.raw))
    if crop is None:
        shape = img_raw.shape
        crop = (0, shape[0], 0, shape[1])
    cropped = img_raw[crop[0]:crop[1], crop[2]:crop[3], :]
    img_1 = preprocess_image(cv2.resize(np.asarray(cropped), (32, 32))).reshape((1, 32, 32, 3))
    img_2 = preprocess_image(cv2.resize(np.asarray(cropped), (32, 32)), convert_to_gray=True).reshape((1, 32, 32, 1))
    prediction = np.argmax((model1.predict(img_1)*0.611319880891668
                            + model2.predict(img_2)*0.19700592970161662
                            + model3.predict(img_1)*0.19167418940671538), axis=1)
    prediction = data['SignName'][prediction].to_string()
    print("Predicted sign: ", prediction)
    plt.imshow(img_raw)
    plt.show()
    plt.imshow(img_1.reshape(32, 32, 3))
    plt.show()
    plt.imshow(img_2.reshape(32, 32), cmap="gray")
    plt.show()
