import tensorflow as tf
import tf2onnx
import onnx


class EasyExporter:
    def __init__(self):
        pass

    def export_to_onnx(self, name, path):
        """
        Function to export from keras to onnx.
        Docs: https://github.com/Gabrli/EasyCNN---docs

        :param name: name for new model
        :param path: path to old model
        """
        model = tf.keras.models.load_model(path)
        model.output_names=['output']
        input_shape = model.input_shape[1:]
        spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
        output_path = name + ".onnx"
        
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
        onnx.save_model(model_proto, output_path)
    

