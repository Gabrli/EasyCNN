from tensorflow.keras.applications import ResNet50, ResNet152, ResNet50V2, ResNet101, ResNet101V2,ResNet152V2, VGG16, VGG19, InceptionV3, MobileNet,MobileNetV2, MobileNetV3Small, InceptionResNetV2, MobileNetV3Large, Xception

def resnet50Preset(classes, x=32, y=32, r=3):
    preset = ResNet50(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def vgg16Preset(classes, x=32, y=32, r=3):
    preset = VGG16(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def vgg19Preset(classes, x=32, y=32, r=3):
    preset = VGG19(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def inceptionV3Preset(classes, x=32, y=32, r=3):
    preset = InceptionV3(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def inceptionResNetV2Preset(classes, x=32, y=32, r=3):
    preset = InceptionResNetV2(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def resnet152Preset(classes, x=32, y=32, r=3):
    preset = ResNet152(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def resnet50v2Preset(classes, x=32, y=32, r=3):
    preset = ResNet50V2(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def resnet101Preset(classes, x=32, y=32, r=3):
    preset = ResNet101(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset


def resnet101v2Preset(classes, x=32, y=32, r=3):
    preset = ResNet101V2(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def resnet152v2Preset(classes, x=32, y=32, r=3):
    preset = ResNet152V2(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def mobilenetPreset(classes, x=32, y=32, r=3):
    preset = MobileNet(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def mobilenetv2Preset(classes, x=32, y=32, r=3):
    preset = MobileNetV2(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def mobilenetv3smallPreset(classes, x=32, y=32, r=3):
    preset = MobileNetV3Small(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset


def mobilenetv3largePreset(classes, x=32, y=32, r=3):
    preset = MobileNetV3Large(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset

def xpectionPreset(classes, x=32, y=32, r=3):
    preset = Xception(include_top=False,input_shape=(x,y,r),pooling='avg',classes=classes,weights='imagenet')
    return preset