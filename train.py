import argparse
import torch
import numpy as np
import os
import pickle
from post_process.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection

from data_loader.one_class_dataset import get_train_dataloader
from model.one_class.models import STPM
from inference import OneClassInference
from utils import reshape_embedding


class OneClassTest():
    def __init__(self, args):
        self.embedding_dir_path = "./embeddings"
        if not os.path.exists(self.embedding_dir_path):
            os.mkdir(self.embedding_dir_path)

        self.embedding_list = []
        self.train_loader = get_train_dataloader(args.dataset_path, args.load_size, args.input_size, args.batch_size)
        self.model = STPM()
        self.model.eval()
        self.inference = OneClassInference(args)

    def test(self):
        for index, batch_data in enumerate(self.train_loader):
            prediction = self.inference.infer(batch_data)
            self.embedding_list.extend(reshape_embedding(np.array(prediction)))
        self.computer_embedding()

    def computer_embedding(self):
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        randomprojector = SparseRandomProjection(n_components='auto',
                                                      eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=randomprojector, already_selected=[],
                                             N=int(total_embeddings.shape[0] * float(args.coreset_sampling_ratio)))
        self.embedding_coreset = total_embeddings[selected_idx]

        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
            pickle.dump(self.embedding_coreset, f)


def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--dataset_path',
                        default=r'/home/changwoo/hdd/datasets/mvtec_anomaly_detection')  # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)  # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--project_root_path',
                        default=r'/home/changwoo/hdd/project_results/patchcore/test')
    parser.add_argument('--method', default=r'NN')
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    one_class_test = OneClassTest(args)
    one_class_test.test()
