import pandas as pd
import os
from load_data import Load_data_image
from svm_model import SVM_Model, LazyClass
import logging
logging.basicConfig(filename="./log.txt",  datefmt='%H:%M:%S')
logging.debug("Debug logging test...")
logger = logging.getLogger('TrafficLightsClassifier')
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    ## Getting the paths to the dataset
    svm = 0
    path = os.getcwd()
    green = r'{}/data/green'.format(path)
    red = r'{}/data/red'.format(path)
    labels = {'green' : 1 , 'red': 0}
    # 2. Load the image data
    # 3. Normalize the dataset
    # Prepare the data to process
    data = Load_data_image(red)
    data_red, labels_red = data.forward()
    data_green = Load_data_image(green)
    data_green, labels_green = data_green.forward()
    data_model = pd.concat((pd.DataFrame(data = data_red),pd.DataFrame(data = data_green)),axis = 0)
    targets = pd.concat((pd.DataFrame(data=labels_red), pd.DataFrame(data = labels_green)), axis = 0)
    logger.info(data_model.shape)
    logger.info(targets.shape)
    if svm == 1:
        svm_model = SVM_Model(data_model, targets)
        trained_model = svm_model.train_model(svm_model, test_size = 0.2)
        y_predictions = svm_model.test_model(trained_model)
        classification_metrics = svm_model.evaluate_model(trained_model, y_predictions)
        svm_model.save_model(trained_model, "svm_traffic_lights")
        logger.info('The performance for the SVM model for traffic lights classification is as following:')
        logger.info(classification_metrics)
    else:
        lazy_class = LazyClass(data_model, targets)
        models, predictions = lazy_class.forward(0.2)
        logger.info('The performance for the SVM model for traffic lights classification is as following:')
        logger.info(models)
        logger.info("These are the predictions of the models:")
        logger.info(predictions)
        
