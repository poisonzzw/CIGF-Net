import argparse
import yaml
import time
from Utils.test1 import Test
def get_parser():
    parser = argparse.ArgumentParser(description='SSC-Train')
    parser.add_argument('--sys_config', type=str, default='./Configs/Voxel_kitti_ssc_sys.yaml')

    args_cfg = parser.parse_args()

    config_sys = yaml.safe_load(open(args_cfg.sys_config, 'r'))

    config_sys['System_Parameters']['time'] = time.strftime('%Yy%mm%dd-%Hh%Mm', time.localtime())
    config_sys['is_test'] = 'False'
    return config_sys

if __name__ == '__main__':
    print('-' * 35)
    print('-----------Start!!!---------------')
    sys_args = get_parser()
    print(sys_args['System_Parameters']['time'])
    test = Test(sys_args=sys_args)
    test.fuse()