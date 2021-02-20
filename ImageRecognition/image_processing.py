import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter


# Decode image data from a file in Tensorflow
def decode_image(filename, image_type, resize_shape, channels=0):
    value = tf.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    if resize_shape is not None and image_type in ['png', 'jpeg']:
        decoded_image = tf.image.resize_images(decoded_image, resize_shape)
    return decoded_image


# Return a dataset created from the image file paths
def get_dataset(image_paths, image_type, resize_shape, channels):
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)

    def _map_fn(filename):
        return ref_decode_image(filename, image_type, resize_shape, channels=channels)

    return dataset.map(_map_fn)


# Get the decoded image data from the input image file paths
def get_image_data(image_paths, image_type=None, resize_shape=None, channels=0):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator = dataset.make_one_shot_iterator()
    next_image = iterator.get_next()
    image_data_list = []
    with tf.Session() as sess:
        for i in range(len(image_paths)):
            image_data = sess.run(next_image)
            image_data_list.append(image_data)
    return image_data_list


# Load and resize an image using PIL, and return its pixel data
def pil_resize_image(image_path, resize_shape, image_mode='RGBA', image_filter=None):
    im = Image.open(image_path)
    converted_im = im.convert(image_mode)
    resized_im = converted_im.resize(resize_shape, Image.LANCZOS)
    if image_filter is not None:
        resized_im = resized_im.filter(image_filter)
    return np.asarray(resized_im.getdata())
