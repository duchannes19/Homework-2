# Used to save plots of accuracy and loss evolution
import matplotlib.pyplot as plt

def save_plot(save_path, history):
    """
        PARAMETERS:
            save_path: The path where the plot will be saved
            history: The history returned by the keras fit function using validation data.

        RETURNS:
            -

        RAISES:
            -
    """

    _, ax = plt.subplots(2, 1, sharex='col', sharey='row')

    ax[0].plot(range(len(history['val_categorical_accuracy'])), history['val_categorical_accuracy'], label='val_accuracy')
    ax[0].plot(range(len(history['categorical_accuracy'])), history['categorical_accuracy'], label='accuracy')
    ax[0].set(ylabel='Accuracy')
    ax[0].legend()

    ax[1].plot(range(len(history['val_loss'])), history['val_loss'], label='val_loss')
    ax[1].plot(range(len(history['loss'])), history['loss'], label='loss')
    ax[1].set(ylabel='Loss')
    ax[1].legend()

    plt.xlabel('Epochs')
    plt.xticks(range(1, len(history['loss']) + 1))

    plt.savefig(save_path, bbox_inches='tight')