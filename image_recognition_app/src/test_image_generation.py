#!/usr/bin/env python3

# For Genrating test images

from PIL import Image
from keras.datasets import mnist
import numpy as np
import os

def generate_random_images(n):
    """
    Generate and save n random images from the MNIST dataset.

    Parameters:
    - n (int): Number of random images to generate and save.

    Notes:
    - The function uses the MNIST dataset to load images.
    - Images are selected randomly from the training set.
    - Each image is saved as a PNG file with the format '{index}.png'.
    """
    (train_images, _), (_, _) = mnist.load_data()
    output_directory = './image_recognition_app/random_images/'

    for i in np.random.randint(0, 10000+1, n):
        arr2im = Image.fromarray(train_images[i])

        image_path = output_directory + '{}.png'.format(i)
        arr2im.save(image_path, "PNG")

generate_random_images(80)