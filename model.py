from tensorflow.keras.layers import (Activation, AveragePooling2D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras as K


def build_pretrained_model(input_shape, output_size, fine_tune_at=None, weights='imagenet', dense_size=64, regularization_factor=0.001, dropout_rate=0.3, base_learning_rate=0.001):
    """
        PARAMETERS:
            -input_shape: The shape of the model's input, e.g. (299, 299, 3)
            -output_size: The number of output classes the model will be trained to classify
            -fine_tune_at[Default: None]: Only valid if weights='imagenet', it represents the number of layers which should be frozen, i.e. not fine-tuned.
                                          It can also take negative values, e.g. -1 -> only the last layer will be left trainable.
                                          If Falsey the whole pretrained model will be frozen
            -weights[Default: 'imagenet']: Should be either imagenet or None for pretrained or nonpretrained xception model.
            -dense_size[Default: 64]: The size of the first dense layer
            -regularization_factor[Default: 0.001]: The parameters of the L2 regularization on the first dense layer
            -dropout_rate[Default: 0.3]: The rate of the dropout layers
            -base_learning_rate[Default: 0.001]: The initial learning rate of the rmsprop optimizer with rho=0.9

        RETURNS:
            Returns a keras model built joining the following models/layers:
                1) Xception model(either pretrained or randomly initialised)
                2) Dense layer with L2
                3) Dropout layer
                4) Pooling layer
                5) Dropout layer
                6) Output (dense) layer

        RAISES:
            -
    """

    # Gather the base model
    base_model = K.applications.xception.Xception(input_shape=input_shape, include_top=False, weights=weights)

    # Freeze a part of or the whole base_model
    if weights == 'imagenet':
        if fine_tune_at:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
        else:
            base_model.trainable = False

    # Complete the model with other layers
    model = K.Sequential([
        base_model,
        K.layers.Dense(dense_size, kernel_regularizer=K.regularizers.l2(regularization_factor), bias_regularizer=K.regularizers.l2(regularization_factor), activation='relu', name='L2_Regularized_Dense'),
        K.layers.Dropout(dropout_rate, name='Dropout_1'),
        K.layers.GlobalAveragePooling2D(name='Pooling'),
        K.layers.Dropout(dropout_rate, name='Dropout_2'),
        K.layers.Dense(output_size, activation='softmax', name='Output_layer')
    ])

    # Compile the model
    model.compile(optimizer=K.optimizers.RMSprop(learning_rate=base_learning_rate, rho=0.9),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


def AlexNet(input_shape, output_size, base_learning_rate):
    # Implementation of AlexNet

    model = K.Sequential()
    
    model.add(Conv2D(filters=64, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
        
    model.add(Conv2D(filters=128, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(output_size, activation='softmax'))

    # Compile
    model.compile(optimizer=K.optimizers.RMSprop(base_learning_rate, rho=0.9),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model
