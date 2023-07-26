import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split,cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score , recall_score , f1_score
import pickle
from lazypredict.Supervised import LazyClassifier


class SVM_Model():

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def prepare_data(self, test_size):
        ## Split the data into train data and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.values, self.targets.values.ravel(), test_size=test_size, shuffle=True)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def svm_gridsearch_model(self):
        ## Support Vector Machine algorithm
        parameters = {'C': [1, 10], 'gamma': [0.001, 0.01, 1], 'kernel' : ['rbf', 'linear']}
        model = SVC()
        grid = GridSearchCV(estimator=model, param_grid=parameters)
        model_trained = grid.fit(self.X_train, self.y_train)
        print(grid)
        # summarize the results of the grid search
        print(grid.best_score_)
        print(grid.best_estimator_)
        return model_trained

    def train_model(self, model, test_size):
        # Train the SVM model using the training data
        self.prepare_data(test_size)
        model.fit(self.X_train, self.y_train)
        print("Mean accuracy of training data:")
        print(model.score(self.X_train, self.y_train))
        print('----------------------------------')
        return model
    
    def test_model(self, model):
        # Test the model on unseen test data
        y_pred = model.predict(self.X_test)
        return y_pred

    def evaluate_model(self, model, y_pred):
        ## Classification metrics
        accuracy = model.score(self.X_test, self.y_test)
        cross_val = np.mean(cross_val_score(model, self.X_test,self.y_test,cv = 5))
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test,y_pred )
        precision = precision_score(self.y_test, y_pred)
        metrics = np.array([accuracy, precision,recall, f1, cross_val])
        metrics = metrics.reshape(1,-1)
        # Create a pandas dataframe returning the classification metrics
        df_metrics = pd.DataFrame(data=metrics, columns=['accuracy', 'precision', 'recall','f1_score', 'cross_validation'])
        return df_metrics
    
    def save_model(self, model, model_name):
        # Save the model having "model_name" 
        with open(f'{model_name}.pkl', 'wb') as mod:
            pickle.dump(model, mod)
    

class LazyClass(SVM_Model):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    def forward(self, test_size):
        models = LazyClassifier(verbose=1, ignore_warnings=True, predictions=True)
        self.prepare_data(test_size)
        models, predictions =  models.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        print(models)
        results = models
        return results, predictions
    

class NeuralNetwork:
    def __init__(self) -> None:
        pass