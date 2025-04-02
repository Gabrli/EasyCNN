from tensorflow.keras import layers, models
import numpy as np
import cv2 as cv
class EasyCNN:
    """
    Docs: https://github.com/Gabrli/EasyCNN---docs
    """
    def __init__(self, input_shape=[32,32,3]):
        """
        :param input_shape: array if three values x, y, r
        """
        self.model = models.Sequential()
        self.input_shape = input_shape
    def add(self,layers):
        """
        Function to adding custom layers by user.
        :param layers: your additional layers
        """
        self.model.add(layers)
    def add_conv(self, filters, kernel_size, activation='relu'):
        if self.model.layers == 0:
            self.model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        self.model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation))
    def add_max_pool(self, pool_size):
        self.model.add(layers.MaxPooling2D((pool_size, pool_size)))
    def add_flatten(self):
        self.model.add(layers.Flatten())
    def add_dense(self, filters, activation='relu'):
        self.model.add(layers.Dense(filters, activation=activation))
    def add_dropout(self, value = 0.3):
        self.model.add(layers.Dropout(value))
    def add_batch_normalization(self):
        self.model.batchNormalization()
    def add_r_flip(self, value="horizontal"):
        self.model.add(layers.RandomFlip(value))
    def add_r_rotate(self, value=0.2):
        self.model.add(layers.RandomRotation(value))
    def add_r_zoom(self, value=0.2):
        self.model.add(layers.RandomZoom(value))
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    def train(self, x, y, test_x, test_y, epochs=10, batch_size=32):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y))    
    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = models.load_model(path)

        
    def predict(self, x):
        img = cv.imread(x)
        img = cv.resize(img, (32,32))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_np = np.array(img)
        img_np = img / 255
        img_np = np.expand_dims(img_np, axis=0)
        predict = self.model.predict(img_np)
        predict_class = np.argmax(predict)
        return predict_class
