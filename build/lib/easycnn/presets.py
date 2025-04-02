from tensorflow.keras.applications import (
ResNet50, ResNet152, ResNet50V2, ResNet101, ResNet101V2, ResNet152V2,
VGG16, VGG19, InceptionV3, MobileNet, MobileNetV2, MobileNetV3Small,
InceptionResNetV2, MobileNetV3Large, Xception
)

class Preset:
    """
        Ready to use preset of keras application.
        Docs: https://github.com/Gabrli/EasyCNN---docs
    """
    def __init__(self):
        self.models = {
            "resnet50": ResNet50,
            "resnet152": ResNet152,
            "resnet50v2": ResNet50V2,
            "resnet101": ResNet101,
            "resnet101v2": ResNet101V2,
            "resnet152v2": ResNet152V2,
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inceptionv3": InceptionV3,
            "inceptionresnetv2": InceptionResNetV2,
            "mobilenet": MobileNet,
            "mobilenetv2": MobileNetV2,
            "mobilenetv3small": MobileNetV3Small,
            "mobilenetv3large": MobileNetV3Large,
            "xception": Xception
        }

    def get_preset(self, model_name, classes, x=32, y=32, r=3):
        """
        Returns a preset model based on the model name.
        
        List of avaliable presets: https://github.com/Gabrli/EasyCNN---docs/blob/main/docs/presets/presets-list.md
        Full Docs: https://github.com/Gabrli/EasyCNN---docs
        
        :param model_name: Name of the model (e.g., 'resnet50', 'vgg16').
        :param classes: Number of output classes.
        :param x: Width of the input image.
        :param y: Height of the input image.
        :param r: Number of channels in the input image.
        :return: A Keras model instance.
        """
        model = self.models.get(model_name.lower())
        if not model:
            raise ValueError(f"Model '{model_name}' is not supported.")
        return model(include_top=False, input_shape=(x, y, r), pooling='avg', classes=classes, weights='imagenet')