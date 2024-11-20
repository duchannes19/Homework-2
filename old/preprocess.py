### USED TO TEST A DIFFERENT RESIZING METHOD ###

# Used to create, process and save the pictures
from PIL import Image

# Used to handle paths consistently with all different OSs
from pathlib import Path

# Used to convert images to arrays
import numpy as np

# Used to parse directories and files
import os

# Used to randomly sample pictures from the datasets
import random

# Used to prevent some warning messages from appearing
import warnings


def get_input_resized_picture(picture, desired_size, output_path=None):
    """
        get_input_resized_picture(picture, desired_size, output_path=None)
        takes as INPUTS:
            -picture: an image file, usually opened via PIL.Image.open(path)
            -desired_size: the size of the square picture we want to fit our resized picture in
            -output_path[Default: None]: the path to store the resized picture, if (not output_path) the picture won't be saved
        DOES:
            Creates a square black image of size desired_size*desired_size,
            pastes on it the resized input picture and returns the result, maybe saving it to output_path in the process.
        and OUTPUTS:
            The resized picture

        DISCLAIMER:
        This code was inspired from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """

    old_size = picture.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = picture.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    if output_path:
        new_im.save(output_path)

    return np.array(new_im)/255


def get_output_label(file, mapping):
    """
        get_output_label(file, mapping)
        takes as INPUTS:
            -file: the path to a given file - every path should be composed like this: /.../subdir/filename
            -mapping: maps the name of the subdirectory of the file to the index of the label of that file
        DOES:
            Nothing particular
        and OUTPUTS:
            The index associated to the label of file.
    """

    try:
        path = str(Path(file))
        return mapping[path[path.rfind('/', 0, path.rfind('/'))+1:path.rfind('/')]]
    except Exception as e:
        print(e)
        raise Exception('The mapping\'s keys miss this subdir key: {}.'.
                        format(path[path.rfind('/', 0, path.rfind('/'))+1:path.rfind('/')]))


def image_generator(dataset_path, desired_size, mapping, batch_size=32, shuffle=True):
    """
        image_generator(dataset_path, desired_size, mapping, batch_size=32, shuffle=True)
        takes as INPUTS:
            -dataset_path: The path to the root folder of the dataset 
            -desired_size: The pictures will be rescaled, using get_input_resized_picture(picture, desired_size)
            -mapping: The labels will be obtained using get_output_label(picture_path, mapping)
            -batch_size[Default = 32]: The desired batch size of the generator
            -shuffle[Default = True]: If true the pictures will be sampled from the dataset in a random order
        DOES:
            It parses all pictures in dataset_path, in particular batch_size pictures at a time
        and OUTPUTS:
            An object shaped as (batch_size, (desired_size, desired_size, 3), 1)
    """

    pictures_paths = list()

    for root, _, files in os.walk(Path(dataset_path)):

        pictures_paths.extend((os.path.join(root, file) for file in files))

    x = list()
    y = list()

    if shuffle:
        random.shuffle(pictures_paths)

    for picture_path in pictures_paths:

        try:
            picture = Image.open(picture_path)
        except IOError:
            continue

        x.append(get_input_resized_picture(picture, desired_size))
        y.append(get_output_label(picture_path, mapping))

        if len(x) == batch_size:

            yield x, y

            x = list()
            y = list()

    yield x, y


if __name__ == "__main__":

    ### -------------------- Resizing training and testing samples -------------------- ###

    training_dataset = './datasets/MWI-Dataset-1.1_2000'
    testing_dataset = './datasets/MWI-Dataset-1.2.5'

    new_training_dataset = './datasets/training'
    new_testing_dataset = './datasets/testing'

    IMG_SIZE = 299

    # Prevents the appearance of messages signaling the presence of non-picture files
    warnings.filterwarnings("ignore")

    for root, subdir, files in os.walk(Path(training_dataset)):
        for file in files:
            try:
                pic = Image.open(os.path.join(root, file))
            except IOError:
                continue

            output_path = os.path.join(new_training_dataset + subdir[subdir.rfind('/'):],
                                       file[:file.rfind('.')] + '_resized' + file[file.rfind('.'):])

            # Saving resized pictures
            get_input_resized_picture(pic, IMG_SIZE, output_path=output_path)

    for root, subdir, files in os.walk(Path(testing_dataset)):
        for file in files:
            try:
                pic = Image.open(os.path.join(root, file))
            except IOError:
                continue

            output_path = os.path.join(new_testing_dataset + subdir[subdir.rfind('/'):],
                                       file[:file.rfind('.')] + '_resized' + file[file.rfind('.'):])

            # Saving resized pictures
            get_input_resized_picture(pic, IMG_SIZE, output_path=output_path)

    # Resets the warning
    warnings.filterwarnings("default")
