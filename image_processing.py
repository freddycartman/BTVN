import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 255

    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image

def apply_average_filter(image, kernel_size=(3, 3)):
    return cv2.blur(image, kernel_size)

def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def apply_max_filter(image, kernel_size=3):
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

def apply_min_filter(image, kernel_size=3):
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))
