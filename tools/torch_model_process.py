import os
import torch
from collections import OrderedDict


class TorchModelProcess():

    def convert_state_dict(self, state_dict):
        """Converts a state dict saved from a dataParallel module to normal
           module state_dict inplace
           :param state_dict is the loaded DataParallel model_state

        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict


    def load_latest_model(self, weight_path, model, dict_name="model"):
        count = self.torchDeviceProcess.getCUDACount()
        checkpoint = None
        if os.path.exists(weight_path):
            try:
                if count > 1:
                    checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
                    state = self.convert_state_dict(checkpoint[dict_name])
                    model.load_state_dict(state)
                else:
                    checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
                    model.load_state_dict(checkpoint[dict_name])
            except Exception as err:
                # os.remove(weight_path)
                checkpoint = None
                EasyLogger.warn(err)
        else:
            EasyLogger.error("Latest model %s exists" % weight_path)