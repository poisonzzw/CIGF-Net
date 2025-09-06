import numpy as np
from Utils.Utils import write_yaml


class SegEvaluator(object):
    def __init__(self, num_class, FLAGS):
        self.FLAGS = FLAGS
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.label_list = ["unlabeled", "car", "person", "bike",
                           "curve", "car_stop", "guardrail", "color_cone", "bump"]

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):  #mAcc
        Acc = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        class_Acc = Acc
        mAcc = np.nanmean(class_Acc)
        return mAcc,class_Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        per_class_iou = MIoU
        MIoU = np.nanmean(MIoU)
        return MIoU, per_class_iou

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_value(self, epoch, writer, LOGS, valid_dataset):
        Pa = self.Pixel_Accuracy()
        mAcc, class_Acc = self.Pixel_Accuracy_Class()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()
        mIoU, per_class = self.Mean_Intersection_over_Union()

        # print(valid_dataset)

        LOGS['Val']['epoch-%03d' % epoch][valid_dataset] = {}
        LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['mIoU'] = str('{:.6f}'.format(mIoU))
        LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['Pa'] = str('{:.6f}'.format(Pa))
        LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['mAcc'] = str('{:.6f}'.format(mAcc))
        LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['FWIoU'] = str('{:.6f}'.format(FWIoU))

        writer.add_scalar('Test-{}/mIoU'.format(valid_dataset), mIoU, epoch + 1)
        writer.add_scalar('Test-{}/Pa'.format(valid_dataset), Pa, epoch + 1)
        writer.add_scalar('Test-{}/mAcc'.format(valid_dataset), mAcc, epoch + 1)
        writer.add_scalar('Test-{}/FWIoU \n'.format(valid_dataset), FWIoU, epoch + 1)

        LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['per_iou'] = {}
        for i, iou in enumerate(per_class):
            LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['per_iou'][self.label_list[i]] = \
                str('{:.6f}'.format(iou))
            writer.add_scalar('{}-(class)/Iou_%s'.format(valid_dataset) % self.label_list[i], iou, epoch + 1)

        LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['per_Acc'] = {}
        for k, acc in enumerate(class_Acc):
            LOGS['Val']['epoch-%03d' % epoch][valid_dataset]['per_Acc'][self.label_list[k]] = \
                str('{:.6f}'.format(acc))
            writer.add_scalar('{}-(class)/Acc_%s'.format(valid_dataset) % self.label_list[k], acc, epoch + 1)

        write_yaml(LOGS, self.FLAGS['Root']['log_path'])
        return mIoU

    def get_testvalue(self):
        Pa = self.Pixel_Accuracy()
        mAcc, class_Acc = self.Pixel_Accuracy_Class()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()
        mIoU, per_class = self.Mean_Intersection_over_Union()

        # print(valid_dataset)



        print("pa:", str('{:.6f}'.format(Pa)))
        print("mAcc:", str('{:.6f}'.format(mAcc)))
        print("FWIoU", str('{:.6f}'.format(FWIoU)))
        print("mIoU", str('{:.6f}'.format(mIoU)))

        print("9类iou：")
        for i, iou in enumerate(per_class):
            print(self.label_list[i],":",str('{:.6f}'.format(iou)))
        print("9类acc：")
        for k, acc in enumerate(class_Acc):
            print(self.label_list[k], ":", str('{:.6f}'.format(acc)))



