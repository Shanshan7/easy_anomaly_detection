from collections import OrderedDict
from tools.model_tool.model_to_onnx import ModelConverter

class OnnxModelConverter():
    def __init__(self, net_config, weight_path, save_dir):
        self.net_config = net_config
        self.weight_path = weight_path
        self.save_dir = save_dir

    def set_convert_param(self, is_convert, input_names, output_names):
        self.is_convert = is_convert
        self.convert_input_names = input_names
        self.convert_output_names = output_names

    def image_model_convert(self, input_size):
        if self.is_convert:
            converter = ModelConverter(input_size)
            self.save_onnx_path = converter.convert_process(self.net_config,
                                                            self.weight_path,
                                                            self.save_dir,
                                                            self.convert_input_names,
                                                            self.convert_output_names)



def test():
    from model.one_class.models import STPM

    model = STPM()
    input_name = ['one_class_input']
    output_name = ['one_class_output']
    onnx_model_converter = OnnxModelConverter(model, "", "one_class.onnx")
    onnx_model_converter.set_convert_param(True, input_name, output_name)
    onnx_model_converter.image_model_convert(input_size=(224, 224))


if __name__ == "__main__":
    test()