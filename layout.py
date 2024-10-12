import tkinter as tk
from tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# Khởi tạo giao diện Tkinter
root = tk.Tk()
root.title('Image Processing Application')

# Chức năng load ảnh
def load_image():
    global img, img_display
    filepath = filedialog.askopenfilename()
    if filepath:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        show_image(img, original_img_label)

# Chức năng hiển thị ảnh
def show_image(image, label):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    label.config(image=image)
    label.image = image

# Tạo nhiễu Gaussian
def add_gaussian_noise():
    global img
    noisy_img = img + np.random.normal(0, 25, img.shape).astype(np.uint8)
    show_image(noisy_img, result_img_label)

# Tạo nhiễu muối tiêu
def add_salt_and_pepper_noise():
    global img
    noisy_img = img.copy()
    salt_pepper_ratio = 0.02
    amount = 0.04
    num_salt = np.ceil(amount * img.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_pepper_ratio))

    # Thêm muối (white pixel)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 255

    # Thêm tiêu (black pixel)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0

    show_image(noisy_img, result_img_label)

# Các bộ lọc
def apply_average_filter():
    global img
    filtered_img = cv2.blur(img, (5, 5))
    show_image(filtered_img, result_img_label)

def apply_median_filter():
    global img
    filtered_img = cv2.medianBlur(img, 5)
    show_image(filtered_img, result_img_label)

def apply_max_filter():
    global img
    kernel = np.ones((5,5), np.uint8)
    filtered_img = cv2.dilate(img, kernel)
    show_image(filtered_img, result_img_label)

def apply_min_filter():
    global img
    kernel = np.ones((5,5), np.uint8)
    filtered_img = cv2.erode(img, kernel)
    show_image(filtered_img, result_img_label)

# Bố trí giao diện
load_button = tk.Button(root, text="Load ảnh", command=load_image)
load_button.grid(row=0, column=0)

gauss_button = tk.Button(root, text="Tạo nhiễu Gauss", command=add_gaussian_noise)
gauss_button.grid(row=0, column=1)

salt_pepper_button = tk.Button(root, text="Tạo nhiễu muối tiêu", command=add_salt_and_pepper_noise)
salt_pepper_button.grid(row=0, column=2)

average_button = tk.Button(root, text="Lọc trung bình", command=apply_average_filter)
average_button.grid(row=1, column=0)

median_button = tk.Button(root, text="Lọc trung vị", command=apply_median_filter)
median_button.grid(row=2, column=0)

max_button = tk.Button(root, text="Lọc MAX", command=apply_max_filter)
max_button.grid(row=3, column=0)

min_button = tk.Button(root, text="Lọc MIN", command=apply_min_filter)
min_button.grid(row=4, column=0)

original_img_label = tk.Label(root)
original_img_label.grid(row=1, column=1, rowspan=4)

result_img_label = tk.Label(root)
result_img_label.grid(row=1, column=2, rowspan=4)

root.mainloop()
