
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils.data_loder import testpath,TESTData
from Utils.metrics import SegEvaluator

class Test(object):
    def __init__(self, sys_args):
        super(Test, self).__init__()
        self.FLAGS = sys_args
    def test_val(self, image_path_a, image_path_b, lable_path):

        # Data
        data = TESTData(image_path_a, image_path_b, lable_path)

        test_data = DataLoader(dataset=data, batch_size=1)

        # BUILD Model

        from Convnextv2.builder import Convnextv2

        pretrain_dict = torch.load('./Checkpoint/BEST-MODAL/checkpoint/Model_statedict.pth', map_location='cpu')
        model = Convnextv2(in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(self.FLAGS['System_Parameters']['device'])
        model_dict = {}
        mstate_dict = model.state_dict()
        for k, v in pretrain_dict['state_dict'].items():
            if k in mstate_dict:
                model_dict[k] = v
        mstate_dict.update(model_dict)
        model.load_state_dict(mstate_dict)

        metric = SegEvaluator(9, self.FLAGS)

        with torch.no_grad():
            model.eval()
            metric.reset()
            with tqdm(total=len(test_data), ncols=100, ascii=True) as t:

                for i, (img_ir, img_vis, img_lable, img_name) in enumerate(test_data):
                    t.set_description('||Test-ALL Image %s' % (i + 1))

                    img_ir = img_ir.to(self.FLAGS['System_Parameters']['device'])
                    img_vis = img_vis.to(self.FLAGS['System_Parameters']['device'])
                    img_lable = img_lable.to(self.FLAGS['System_Parameters']['device']).long()

                    outputs = model(img_vis, img_ir)

                    metric_predict = outputs.softmax(dim=1).argmax(dim=1)
                    metric.add_batch(img_lable.detach().cpu().numpy(), metric_predict.detach().cpu().numpy())

                    t.update(1)
                metric.get_testvalue()

    def fuse(self):
        image_path_a, image_path_b, lable_path = testpath()
        self.test_val(image_path_a, image_path_b, lable_path)


