import cv2
import numpy as np
from random import randint, uniform
import os


def random_crop(img):
    height, width, channels = img.shape
    top_crop = randint(35, int(height / 8))
    bottom_crop = height - randint(20, int(height / 10))
    left_crop = randint(35, int(width / 8))
    right_crop = width - randint(20, int(width / 10))
    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]
    return crop_img


def blur(img, kernel_size):
    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur_img


def random_rotate(img):
    negative_angle = randint(-20, -10)
    positive_angle = randint(10, 20)
    if abs(negative_angle) >= positive_angle:
        angle = negative_angle
    else:
        angle = positive_angle
    height = img.shape[0]
    width = img.shape[1]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    return rotated_img


def random_rotate_zoom(img):
    negative_angle = randint(-20, -10)
    positive_angle = randint(10, 20)
    if abs(negative_angle) >= positive_angle:
        angle = negative_angle
    else:
        angle = positive_angle
    zoom = uniform(1.1, 1.4)
    height = img.shape[0]
    width = img.shape[1]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.2)
    rotated_zoomed_img = cv2.warpAffine(img, M, (width, height))
    return rotated_zoomed_img


def random_translation_rotation(img):
    height, width, channels = img.shape
    vertical_translation = randint(- int(height / 25), int(height / 25))
    horizontal_translation = randint(- int(width / 25), int(width / 25))
    M = np.float32([[1, 0, horizontal_translation],
                    [0, 1, vertical_translation]])
    dst = cv2.warpAffine(img, M, (width, height))
    rotated_translated_img = random_rotate(dst)
    return rotated_translated_img


def affine_transformation(img):
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations
    pass


def jittering(img):
    height, width, channels = img.shape
    noise = np.random.randint(0, 35, (height, width))
    zitter = np.zeros_like(img)
    zitter[:, :, 1] = noise
    jittered_img = cv2.add(img, zitter)
    return jittered_img


if __name__ == "__main__":
    absolute_path = 'assets/posters/'
    for folder in os.listdir(absolute_path):
        print(folder)
        for filename in os.listdir(absolute_path + folder):
            print(absolute_path + folder)
            print(filename)
            image_path = absolute_path + folder + '/' + filename
            img = cv2.imread(image_path)
            crop_img = random_crop(img)
            # crop_translated_img = random_crop_translation(img)
            cv2.imwrite(absolute_path + folder + '/' +
                        "crop" + filename, crop_img)
            print(absolute_path + folder + '/' + "crop" + filename)
            # cv2.imwrite('crop_translated' + image_path, crop_translated_img)
            blur_5 = blur(img, 5)
            blur_7 = blur(img, 7)
            cv2.imwrite(absolute_path + folder + '/' +
                        'blur_5' + filename, blur_5)
            cv2.imwrite(absolute_path + folder + '/' +
                        'blur_7' + filename, blur_7)
            rotated_img = random_rotate(img)
            cv2.imwrite(absolute_path + folder + '/' +
                        'rotated' + filename, rotated_img)
            zoom_rotated_img_1 = random_rotate_zoom(img)
            zoom_rotated_img_2 = random_rotate_zoom(img)
            cv2.imwrite(absolute_path + folder + '/' +
                        'zoom_rotated_1' + filename, zoom_rotated_img_1)
            cv2.imwrite(absolute_path + folder + '/' +
                        'zoom_rotated_2' + filename, zoom_rotated_img_2)
            translated_rotated_img_1 = random_translation_rotation(img)
            translated_rotated_img_2 = random_translation_rotation(img)
            cv2.imwrite(absolute_path + folder + '/' +
                        'translated_rotated_1' + filename, translated_rotated_img_1)
            cv2.imwrite(absolute_path + folder + '/' +
                        'translated_rotated_2' + filename, translated_rotated_img_2)
            jittered_img = jittering(img)
            cv2.imwrite(absolute_path + folder + '/' +
                        'jittered_img' + filename, jittered_img)
        break
