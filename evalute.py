import numpy as np
from sklearn.metrics import roc_auc_score


class OneClassEvaluate():
    def __init__(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

        self.fp_rate = 0.0
        self.pixel_auc = 0.0
        self.img_auc = 0.0

    def eval(self):
        gt_list = np.asarray(self.gt_list_img_lvl)
        img_scores = np.asarray(self.pred_list_img_lvl)
        threshold_ng = min(img_scores[gt_list==1])
        ng_scores = img_scores[gt_list==0]
        ng_scores[ng_scores>threshold_ng] = 10
        ng_scores[ng_scores<threshold_ng] = 0
        self.fp_rate = sum(ng_scores) / sum(1-gt_list) / 10

        print("Total image-level auc-roc score :")
        self.img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        # print("Total pixel-level auc-roc score :")
        # self.pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)

    def get_score(self):
        score = {'fp_rate': self.fp_rate,
                 # 'pixel_auc': self.pixel_auc,
                 'img_auc': self.img_auc}
        self.print_evaluation(score)
        return score

    def print_evaluation(self, score):
        for k, v in score.items():
            print("{}:{}".format(k, v))

    # def AUCPRO(self):
    #     max_step = 1000
    #     expect_fpr = 0.3  # default 30%
    #     max_th = super_mask.max()
    #     min_th = super_mask.min()
    #     delta = (max_th - min_th) / max_step
    #     ious_mean = []
    #     ious_std = []
    #     pros_mean = []
    #     pros_std = []
    #     threds = []
    #     fprs = []
    #     binary_score_maps = np.zeros_like(super_mask, dtype=np.bool)
    #     for step in range(max_step):
    #         thred = max_th - step * delta
    #         # segmentation
    #         binary_score_maps[super_mask <= thred] = 0
    #         binary_score_maps[super_mask >  thred] = 1
    #         pro = []  # per region overlap
    #         iou = []  # per image iou
    #         # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
    #         # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
    #         for i in range(len(binary_score_maps)):    # for i th image
    #             # pro (per region level)
    #             label_map = label(gt_mask[i], connectivity=2)
    #             props = regionprops(label_map)
    #             for prop in props:
    #                 x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region
    #                 cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
    #                 # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
    #                 cropped_mask = prop.filled_image    # corrected!
    #                 intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
    #                 pro.append(intersection / prop.area)
    #             # iou (per image level)
    #             intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
    #             union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
    #             if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
    #                 iou.append(intersection / union)
    #         # against steps and average metrics on the testing data
    #         ious_mean.append(np.array(iou).mean())
    #         #print("per image mean iou:", np.array(iou).mean())
    #         ious_std.append(np.array(iou).std())
    #         pros_mean.append(np.array(pro).mean())
    #         pros_std.append(np.array(pro).std())
    #         # fpr for pro-auc
    #         gt_masks_neg = ~gt_mask
    #         fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
    #         fprs.append(fpr)
    #         threds.append(thred)
    #     # as array
    #     threds = np.array(threds)
    #     pros_mean = np.array(pros_mean)
    #     pros_std = np.array(pros_std)
    #     fprs = np.array(fprs)
    #     ious_mean = np.array(ious_mean)
    #     ious_std = np.array(ious_std)
    #     # best per image iou
    #     best_miou = ious_mean.max()
    #     print(f"Best IOU: {best_miou:.4f}")
    #     # default 30% fpr vs pro, pro_auc
    #     idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    #     fprs_selected = fprs[idx]
    #     fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    #     pros_mean_selected = pros_mean[idx]
    #     seg_pro_auc = auc(fprs_selected, pros_mean_selected)
    #     seg_pro_obs.update(100.0*seg_pro_auc, epoch)

# def test_epoch_end(self, outputs):
