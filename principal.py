from tools.image_classification_pipeline import ImageProcessing


if __name__ == "__main__":
    labels_map = {0 : 'green', 1: 'red'}
    path_images = "./data"
    image_pipeline = ImageProcessing()
    images, labels = image_pipeline.process_images(path = path_images, labels_map=labels_map, display = True)