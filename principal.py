from tools.image_classification_pipeline import ImageProcessing


if __name__ == "__main__":
    image_pipeline = ImageProcessing()
    images, labels = image_pipeline.processed_images("./data", display = True)