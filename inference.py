import argparse
import torch
import time
import numpy as np
import os
import pickle

from models import STPM
from data_loader import get_test_dataloader
from evalute import OneClassEvaluate
from utils import reshape_embedding
from sampling_methods.KNN import KNN
from sklearn.neighbors import NearestNeighbors


THRESHOLD = 1.0


class OneClassInference():
    def __init__(self, args):
        super().__init__()
        self.test_loader = get_test_dataloader(args.dataset_path, args.load_size, args.input_size)
        self.one_class_evaluate = OneClassEvaluate()
        self.n_neighbors = args.n_neighbors
        self.model = STPM()
        self.model.eval()
        self.method = args.method

    def process(self):
        for index, batch_data in enumerate(self.test_loader):
            start_time = time.time()
            prediction = self.infer(batch_data)
            class_index, class_confidence = self.post_process(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, time.time() - start_time))

            self.one_class_evaluate.gt_list_img_lvl.append(batch_data[2].cpu().numpy()[0])
            self.one_class_evaluate.pred_list_img_lvl.append(class_confidence)
        self.one_class_evaluate.eval()
        self.one_class_evaluate.get_score()

    def infer(self, batch_data):
        x, gt, label, file_name, x_type = batch_data
        # extract embedding
        # print("x: {}".format(x.sum()))
        prediction = self.model(x, 1)
        # print("output: {}".format(prediction))
        return prediction

    def post_process(self, prediction):
        embedding_coreset = pickle.load(open(os.path.join("./embeddings", 'embedding.pickle'), 'rb'))
        embedding_test = np.array(reshape_embedding(np.array(prediction)))

        if self.method == "KNN":
            knn = KNN(torch.from_numpy(embedding_coreset).cuda(), k=self.n_neighbors)
            score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()
        else:
            nbrs = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(
                embedding_coreset)
            score_patches, _ = nbrs.kneighbors(embedding_test)

        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])  # Image-level score
        if score > THRESHOLD:
            class_index = 0
        else:
            class_index = 1

        return class_index, score

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--dataset_path',
                        default=r'/home/changwoo/hdd/datasets/mvtec_anomaly_detection')  # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--load_size', default=256)  # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--project_root_path',
                        default=r'/home/changwoo/hdd/project_results/patchcore/test')
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--method', default=r'NN')
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    one_class_test = OneClassInference(args)
    one_class_test.process()

