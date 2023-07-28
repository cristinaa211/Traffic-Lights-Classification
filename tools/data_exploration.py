from torchvision import transforms
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_random_images(images, labels, labels_map):
    random_indexs = random.sample(range(0, len(images)), 9)
    fig = plt.figure(figsize=(16, 10))
    for idx in random_indexs:
        subplot_index = random_indexs.index(idx)
        image = transforms.ToPILImage()(images[idx])
        plt.subplot(3, 3, subplot_index+1)
        plt.imshow(image)
        plt.title(labels_map[labels[idx]])
    plt.show()


def plot_image_histogram(image):
    color = ('b', 'g', 'r')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    for idx, col in enumerate(color):
        histogram = cv2.calcHist([image], [idx], None, [256], [0, 256])
        histogram = normalize_histogram(histogram)
        plt.subplot(1, 2, 2)
        plt.plot(histogram, color = col)
        plt.xlabel('Color value')
        plt.ylabel('Frequency')
        plt.title('Image Histogram')
        plt.xlim([0, 256])
    plt.show()

def histogram_image(img, mask = None, size = 256):
    hist_blue = cv2.calcHist(images = [img], channels = [0], mask = mask, histSize = [size],ranges = [0, size]) 
    hist_green = cv2.calcHist(images = [img],channels = [1],mask = mask, histSize = [size],ranges = [0, size]) 
    hist_red = cv2.calcHist(images = [img], channels = [2],mask = mask, histSize = [size],ranges = [0, size]) 
    hist_blue_norm = normalize_histogram(hist_blue)
    hist_red_norm = normalize_histogram(hist_red)
    hist_green_norm = normalize_histogram(hist_green)
    return hist_blue_norm, hist_green_norm, hist_red_norm

def normalize_image(image):
    return cv2.normalize(image, image, 0, 1.0, cv2.NORM_MINMAX)

def normalize_histogram(hist):
    return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

def extract_edges(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, threshold1 = 100, threshold2 = 500)
    return edges

def rgb_hsv_conversion(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.set_title('Standardized image')
    ax1.imshow(image)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')
    plt.show()
    return h, s, v


if __name__ == "__main__":
    image = "./data/green/abc016.jpg"
    image = cv2.imread(image)
    rgb_hsv_conversion(image)