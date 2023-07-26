import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess
import time



def plot_image_histogram(image):
    image = cv2.imread(image)
    color = ('b', 'g', 'r')
    plt.figure()
    for idx, col in enumerate(color):
        histogram = cv2.calcHist([image], [idx], None, [256], [0, 256])
        histogram = normalize_histogram(histogram)
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
    # plt.show()
    return h, s, v



def object_detection(image_path, yolo_path):
    command = ['bash', '-c', '{} detect cfg/yolov3.cfg yolov3.weights {}'.format(yolo_path, image_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE,  shell = True)
    output, err = process.communicate()
    p_status = process.wait()
    return output.decode(), p_status

def object_detection_yolo(img, yolo_path):
    img = cv2.imread(img)
    config_file = f"{yolo_path}/cfg/yolov3.cfg"
    weights_file = f"{yolo_path}/yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getLayerNames()
    print(len(ln), ln)
    # blob_false = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    blob_false = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=True)
    r = blob_false[0, 0, :, :]
    cv2.imshow('blob', r)
    text = f'Blob shape={blob_false.shape}'
    cv2.displayOverlay('blob', text)
    cv2.waitKey(1)
    net.setInput(blob_false)
    t0 = time.time()
    outputs = net.forward(ln)
    print(outputs)
    t = time.time()
    cv2.displayOverlay('window', f'forward propagation time={t-t0}')
    cv2.imshow('window',  img)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_features_pipeline(image_path):
    image = cv2.imread(image_path)
    edges = extract_edges(image)
    h, s, v = rgb_hsv_conversion(image)
    norm_img = normalize_image(image)
    hist_blue_norm, hist_green_norm, hist_red_norm = histogram_image(norm_img)


if __name__ == '__main__':
    img = '/home/cristina/Documents/github/Traffic_lights_classification/data/green/green_106.jpg'
    yolo_path = "/home/cristina/Documents/darknet"
    # result = object_detection(img, yolo_path)
    height, width, channels = cv2.imread(img).shape
    print(f'height = {height}, width = {width}, channels = {channels}')
    # plot_image_histogram(img)
    object_detection_yolo(img, yolo_path)
    print('ok')