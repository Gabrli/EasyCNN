
# EasyCNN - Easy Creating and Visualizing CNN Model

EasyCNN is a Python library designed to simplify the creation, training, and visualization of convolutional neural networks (CNNs). It provides an intuitive interface for deep learning enthusiasts and developers who want to work with CNN models without the complexity often associated with neural network frameworks.


## Features

- Straightforward definition of CNN layers with intuitive syntax
- Streamlined training and model evolution capabilities
- Visual representation of model architecture
- 15 pre-configured popular Keras application models ready for immediate use
- Seamless conversion of Keras models to ONNX format



## Documentation

[Documentation](https://github.com/Gabrli/EasyCNN---docs)



## Authors

- [@Gabrli](https://github.com/Gabrli)


## Tech Stack

**Languages:** Python
  
**Libraries:** Tensorflow, Matplotlib, Numpy, OpenCv

## License

[MIT](https://choosealicense.com/licenses/mit/)


## FAQ

#### what are the advantages ?

- Very easy and comfortable syntax
- Full control for developer
- Automatic data preparation and visualization processes
- Compatibility of model: option to convert to onnx type file.

#### What functionalities are under construction?

- Presets for popular models
- Exporter and Converter for files with models
- Special Visualizer to display training process


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Basic Usage/Example

```python
from easycnn.core import EasyCNN
from easycnn.visualizer import TrainingVisualizer
import os
from tensorflow.keras.datasets import cifar10

class_names = ['car', 'plane', 'cat', 'dog', 'bird', 'deer', 'horse', 'frog', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

my_file = os.path.join(os.path.dirname(__file__), 'car.jpg')

x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:400]
y_test = y_test[:400]
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
history = model.train(x_train, y_train, x_test, y_test, epochs=5)
prediction = model.predict(my_file)

visualizer = TrainingVisualizer()
visualizer.plot_training(history)
```

