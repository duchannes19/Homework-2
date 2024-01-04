### Generate a .csv file for the blindtestdataset

# Used to load the saved model
from tensorflow.keras.models import load_model

# Used to open and resize input images
from PIL import Image

# Needed for np.argmin
import numpy as np

# Used to navigate the files
import os

if __name__ == "__main__":

    blindtest_path = './WeatherBlindTestSet'
    csv_path = './1711234.csv'

    mapping = {0: 'HAZE\n', 1: 'RAINY\n', 2: 'SNOWY\n', 3: 'SUNNY\n'}
    fd = open('/content/gdrive/My Drive/Unsynced/1711234.csv', mode='w')
    
    loaded_model = load_model('./1711234.h5')

    for root, _, files in os.walk(blindtest_path):
        for f in files:
            try:
                pic = Image.open(os.path.join(root, f)).convert('RGB')
                pic = pic.resize((299, 299), Image.ANTIALIAS)
                fd.write(mapping[np.argmax(loaded_model.predict(np.reshape(np.array(pic), (1, 299, 299, 3))))])
            except IOError:
                print("problem")

    fd.close()
