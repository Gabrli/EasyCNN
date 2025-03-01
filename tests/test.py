from easycnn.core import EasyCNN
import os
from tensorflow.keras.datasets import cifar10

class_names = ['car', 'plane', 'cat', 'dog', 'bird', 'deer', 'horse', 'frog', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

my_file = os.path.join(os.path.dirname(__file__), 'car.jpg')


x_train = x_train[:20000]
y_train = y_train[:20000]
x_test = x_test[:4000]
y_test = y_test[:4000]
x_train = x_train / 255
x_test = x_test / 255

model = EasyCNN()
model.add_conv(32, 3)
model.add_max_pool(2)
model.add_conv(64, 3)
model.add_max_pool(2)
model.add_conv(128, 3)
model.add_max_pool(2)
model.add_flatten()
model.add_dense(10, activation='softmax')
model.compile()
model.train(x_train, y_train, x_test, y_test, epochs=10)
prediction = model.predict(my_file)

model.save('mymodel.h5')

print(class_names[prediction])