import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, Add, PReLU
from tensorflow.python.keras.layers import BatchNormalization, GlobalAveragePooling2D, Concatenate, GlobalMaxPool2D
from tensorflow.python.keras.regularizers import L1L2
from my_metrics.metrics import f1_score


def efficient_model(filters=48, input_shape=(32, 32, 3), regularizers=(0, 1e-5), num_classes=43,
                    prefix="", loss="categorical_crossentropy"):
    """
    This function is used to create the network.

    Args:
         filters (int): maximum number of filters used in the convolution layers of this network
         input_shape (tuple or list): the input shape of the network
         regularizers(tuple or list): L1 and L2 alpha values respectively
         num_classes (int): the number of classes
         prefix (str): to avoid the problem of having a unique layer name
         loss (str or loss function): the loss function to use

    Returns:
        model (tf.keras.Model): the network already compiled and ready to use for training
    """

    def conv_batch_prelu(name, tensor, num_filters, kernel_size=(3, 3), strides=(1, 1), padding="same"):
        """
        This function combines conv2d layer, batch normalization layer and prelu activation.

        Args:
            name (str): layer's name ('conv_', 'batchnorm' and 'prelu' are added to the name)
            tensor (tf.Tensor): the input tensor
            num_filters (int): number of filters used in the convolution layer
            kernel_size (tuple or list): size of each kernel in the convolution
            strides (tuple or list): strides used in the convolution
            padding (str): one of 'same' or 'valid'

        Return:
            tensor (tf.Tensor): the output tensor
        """
        tensor = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                        kernel_initializer="he_uniform", bias_initializer="zeros",
                        kernel_regularizer=L1L2(regularizers[0], regularizers[1]),
                        padding=padding, name=f"{prefix}_conv_{name}")(tensor)
        tensor = BatchNormalization(momentum=0.1, name=f"{prefix}_batchnorm_{name}")(tensor)
        tensor = PReLU(shared_axes=[1, 2], name=f"{prefix}_prelu_{name}")(tensor)
        return tensor

    def dense_batch_prelu(name, tensor, n_units):
        """
        This function combines dense layer, batch normalization layer and prelu activation.

        Args:
            name (str): layer's name ('dense_', 'batchnorm' and 'prelu' are added to the name)
            tensor (tf.Tensor): the input tensor
            n_units (int): number of units in the dense layer

        Return:
            tensor (tf.Tensor): the output tensor
        """
        tensor = Dense(n_units, name=f"{prefix}_dense_{name}", kernel_initializer="he_uniform",
                       bias_initializer="zeros")(tensor)
        tensor = BatchNormalization(momentum=0.1, name=f"{prefix}_batchnorm_{name}")(tensor)
        tensor = PReLU(name=f"{prefix}_prelu_{name}")(tensor)
        return tensor

    # input layer
    inp = Input(shape=input_shape, name=f"{prefix}_input")

    # 1st convolution block
    cbp1 = conv_batch_prelu("cbp1", inp, num_filters=filters, kernel_size=(9, 9), strides=(4, 4))

    # 2nd convolution block
    cbp2 = conv_batch_prelu("cbp2", inp, num_filters=filters//2, kernel_size=(5, 5), strides=(2, 2))

    # 3rd convolution block
    cbp3 = conv_batch_prelu("cbp3", inp, num_filters=filters//2, kernel_size=(5, 5), padding='Same')
    cbp4 = conv_batch_prelu("cbp4", cbp3, num_filters=filters//2, kernel_size=(5, 5), padding='Same')
    max_pool1 = MaxPool2D(pool_size=(2, 2), name=f"{prefix}_max_pool1")(cbp4)

    # 1st concatenation
    concatenate1 = Concatenate(name=f"{prefix}_concatenate1")([cbp2, max_pool1])

    # 4th convolution block
    cbp5 = conv_batch_prelu("cbp5", concatenate1, num_filters=filters, kernel_size=(5, 5), strides=(2, 2))

    # 5th convolution block
    cbp6 = conv_batch_prelu("cbp6", concatenate1, num_filters=filters, kernel_size=(5, 5), padding='Same')
    cbp7 = conv_batch_prelu("cbp7", cbp6, num_filters=filters, kernel_size=(5, 5), padding='Same')
    max_pool2 = MaxPool2D(pool_size=(2, 2), name=f"{prefix}_max_pool2")(cbp7)

    # 2nd concatenation
    concatenate2 = Concatenate(name=f"{prefix}_concatenate2")([cbp5, max_pool2, cbp1])

    # 1st fully connected
    avg_pool = GlobalAveragePooling2D(name=f"{prefix}_avg_pool")(concatenate2)
    fc1 = dense_batch_prelu("fc1", avg_pool, 1024)

    # 2nd fully connected
    global_pool = GlobalMaxPool2D(name=f"{prefix}_global_pool")(concatenate2)
    fc2 = dense_batch_prelu("fc2", global_pool, 1024)

    # combine
    add = Add(name=f"{prefix}_add")([fc1, fc2])
    drop1 = Dropout(0.5, name=f"{prefix}_drop1")(add)

    # 3rd fully connected
    fc3 = dense_batch_prelu("fc3", drop1, 512)
    drop2 = Dropout(0.5, name=f"{prefix}_drop2")(fc3)

    # output
    out = Dense(num_classes, activation="softmax", name=f"{prefix}_output")(drop2)

    # compile model
    model = tf.keras.Model(inp, out)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", f1_score])

    return model
