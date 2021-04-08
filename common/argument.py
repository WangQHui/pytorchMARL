import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # 环境参数设置
    parser.add_argument('--difficulty', type=str, default='7', help='the ')