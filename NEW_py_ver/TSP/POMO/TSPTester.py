
import torch

import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        self.register_last()
        # Restore
        checkpoint_path = self.env_params['model_path']
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)

        # utility
        self.time_estimator = TimeEstimator()

    def run_local_dir(self):
        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        if not os.path.exists(self.env_params['local_dir']): return
        local_dir = self.env_params['local_dir']
        local_paths = os.listdir(local_dir)
        all_path_scores = []
        all_path_aug_scores = []
        for local_path in local_paths:
            print(local_path)
        for local_path in local_paths:
            local_path = local_dir + local_path
            batch_size = self.tester_params['test_batch_size']
            score, aug_score = self._test_one_batch(batch_size, local_path)
            score = score * (self.env.max_x - self.env.min_x) + self.env.min_x
            print(local_path, score)
            all_path_scores.append(score)
        return local_paths, all_path_scores
        # print(all_path_scores)

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                return score_AM.avg

    def _test_one_batch(self, batch_size, local_path=None):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor, local_path)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()


    def register_last(self):
        # save external param as last
        last_k = []
        last_b = []
        for param_name, param in self.model.encoder.layers.named_parameters():
            if 'external_k' in param_name:
                last_k.append(param.data.clone())
            if 'external_b' in param_name:
                last_b.append(param.data.clone())
        last_k = torch.mean(torch.stack(last_k, dim=0), dim=0)
        last_b = torch.mean(torch.stack(last_b, dim=0), dim=0)
        self.model.register_buffer('last_k', last_k)
        self.model.register_buffer('last_b', last_b)
        self.model.register_buffer('grad_k', last_k)
        self.model.register_buffer('grad_b', last_b)
