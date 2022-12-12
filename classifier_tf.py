import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
import yaml
import logging
log = logging.getLogger(__name__)


def read_yaml(file_path):
    """ Read yaml config file """
    log.info("Reading yaml file: {}".format(file_path))
    with open(file_path, "r") as f:
        return yaml.safe_load(f)



class ClassifierTFModel():
    def __init__(self, config_file):
        """
        The classifier model class should include two methods
        train() and predict() 
        """
        self.conf = read_yaml(config_file)


    def _build_model(self, X_dim, y_dim):
        """ Using Functional Model API """

        layers_ = self.conf['tf']['layers_']
        
        X = Input(shape = (X_dim,))

        # First layer requires input X
        x = Dense(layers_[0], activation = self.conf['tf']['activ'])(X)
        
        if len(layers_)>1:
            for l_ in layers_[1:]:
                x = Dense(l_, activation = self.conf['tf']['activ'])(x)
        # use sigmoid activation to bound result between 0 to 1
        y = Dense(y_dim, activation="sigmoid")(x)
    
        return Model(inputs=X, outputs=y, name="Classifier")
        

    
    def train(self, X_train, X_test, y_train, y_test):
        """
        Train model. 
        The model is build before training 
        because tensorflow requires access to input/output size
        
        """
        X_dim = X_train.shape[-1]
        y_dim = y_train.shape[-1]
        self.model = self._build_model(X_dim, y_dim)
        self.model.compile(optimizer=self.conf['tf']['optimizer'],
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
        history=self.model.fit(
            X_train, 
            y_train, 
            epochs=self.conf['tf']['epochs'], 
            batch_size=self.conf['tf']['n_batch'])

        loss, self.score = self.model.evaluate(X_test, y_test)


    def predict(self, X_inference, return_labels=False):
        """ Inference and returns result """
        return self.model.predict(X_inference)