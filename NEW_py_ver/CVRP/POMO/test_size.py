##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester_size import CVRPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'type': 'ou',
    'model_path': 'models/model_continue.pt',
    'local_dir': '../tsplib_npy/'
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_CVRP100_model',  # directory path of pre-trained model and log files saved.
        'epoch': 30500,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        'filename': '../vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)

    return tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    with open('res_size.txt', 'a') as f:
        f.write(env_params['model_path'] + '\n')
        for size in [20, 40, 60, 80, 100, 150, 200, 300, 400, 500]:
            env_params['problem_size'] = size
            env_params['pomo_size'] = size
            f.write(str(size) + ' ' + str(main()) + '\n')