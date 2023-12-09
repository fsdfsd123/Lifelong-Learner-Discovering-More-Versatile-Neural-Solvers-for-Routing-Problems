
import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from torch import autograd
device_ids = list(range(8))

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 init = False
                 ):

        # save arguments
        self.init = init
        self.env_params = env_params
        self.pomo_size = env_params['pomo_size']
        self.problem_size = env_params['problem_size']
        self.mix_env_params = env_params['mix_env_params']
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        base_episode = 1000
        self.train_num_episode_params = [{0: 100 * base_episode},
                                        {0: 20 * base_episode, 1:100 * base_episode},
                                        {0: 20*base_episode, 1: 20*base_episode, 2: 100*base_episode},
                                        {0: 20*base_episode, 1: 20*base_episode, 2: 20*base_episode, 3: 100*base_episode},
                                        {0: 20*base_episode, 1: 20*base_episode, 2: 20*base_episode, 3: 20*base_episode, 4: 100*base_episode},
                                        {0: 20*base_episode, 1: 20*base_episode, 2: 20*base_episode, 3: 20*base_episode, 4: 20*base_episode, 5: 100*base_episode},
                                        {0: 20*base_episode, 1: 20*base_episode, 2: 20*base_episode, 3: 20*base_episode, 4: 20*base_episode, 5: 20*base_episode, 6: 100*base_episode},
                                        ]

        self.train_num_episode_params = self.train_num_episode_params[self.env_params['mix_num']-1]
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)

        self.envs = [Env(**self.mix_env_params[i]) for i in range(3)]
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.start_epoch = 1
        model_load = trainer_params['model_load']

        self.register_last()
        if os.path.exists(self.env_params['model_path']):
            self.model.load_state_dict(torch.load(self.env_params['model_path']))
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.register_last()
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            for env_id in range(self.env_params['mix_num']):

                self.logger.info('=================================================================')

                # LR Decay
                self.scheduler.step()

                # Train
                train_num_episode = self.train_num_episode_params[env_id]
                print('env_id:', env_id, 'train_num_episode:', train_num_episode)
                train_score, train_loss = self._train_one_epoch(epoch, env_id)
                self.result_log.append('train_score', epoch, train_score)
                self.result_log.append('train_loss', epoch, train_loss)

                ############################
                # Logs & Checkpoint
                ############################
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch,
                                                                                       self.trainer_params['epochs'])
                self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

                all_done = (epoch == self.trainer_params['epochs'])
                model_save_interval = self.trainer_params['logging']['model_save_interval']
                img_save_interval = self.trainer_params['logging']['img_save_interval']

                if epoch > 1:  # save latest images, every epoch
                    self.logger.info("Saving log_image")
                    image_prefix = '{}/latest'.format(self.result_folder)
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                   self.result_log, labels=['train_score'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['train_loss'])

                if all_done or (epoch % model_save_interval) == 0:
                    self.logger.info("Saving trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

                if all_done or (epoch % img_save_interval) == 0:
                    image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                   self.result_log, labels=['train_score'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['train_loss'])

                if all_done:
                    self.logger.info(" *** Training Done *** ")
                    self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                torch.save(self.model.state_dict(), self.env_params['model_path'])

    def _train_one_batch(self, batch_size, env_id):

        # Prep
        ###############################################
        self.model.train()
        self.envs[env_id].load_problems(batch_size)
        reset_state, _, _ = self.envs[env_id].reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.envs[env_id].pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.envs[env_id].pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.envs[env_id].step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()
        # print('loss_mean', loss_mean)
        # if not self.init:
        # grad_k, grad_b, last_k, last_b = [], [], [], []
        current_k, current_b = [], []

        for param_name, param in self.model.encoder.layers.named_parameters():
            if 'external_k' in param_name:
                current_k.append(param)
            if 'external_b' in param_name:
                current_b.append(param)
        current_k = torch.mean(torch.stack(current_k, dim=0), dim=0)
        current_b = torch.mean(torch.stack(current_b, dim=0), dim=0)
        grad_k = None
        for name, param in self.model.named_buffers():

            if 'grad_k' or 'grad_b' in name:
                # print('life long')
                life_long = 1
            if 'grad_k' in name:
                grad_k = param
            if 'grad_b' in name:
                grad_b = param
            if 'last_k' in name:
                last_k = param
            if 'last_b' in name:
                last_b = param
        if grad_k is not None and grad_b is not None:
            life_long_loss = torch.norm(grad_k * loss_mean * (current_k - last_k), p=1) + torch.norm(
                grad_b * loss_mean * (current_b - last_b), p=1)
            if self.init:
                loss_mean += life_long_loss
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        # Step & Return
        ###############################################

        self.model.zero_grad()
        loss_mean.backward()
        mean_grad_k, mean_grad_b = self.get_layers_mean_grad()
        self.optimizer.step()

        return score_mean.item(), loss_mean, mean_grad_k, mean_grad_b

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

    def get_layers_mean_grad(self):
        mean_grad_k, mean_grad_b = [], []
        for name, param in self.model.encoder.layers.named_parameters():
            if 'external_k' in name:
                if param.grad is not None:
                    mean_grad_k.append(param.grad)
            if 'external_b' in name:
                if param.grad is not None:
                    mean_grad_b.append(param.grad)
        if len(mean_grad_k) != 0 and len(mean_grad_b) != 0:
            mean_grad_k = torch.mean(torch.stack(mean_grad_k), dim=0)
            mean_grad_b = torch.mean(torch.stack(mean_grad_b), dim=0)
        return mean_grad_k, mean_grad_b

    def _train_one_epoch(self, epoch, env_id):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.train_num_episode_params[env_id]
        mix_num = self.env_params['mix_num']
        episode = 0
        loop_cnt = 0
        grad_k = []
        grad_b = []

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, loss_mean, mean_grad_k, mean_grad_b = self._train_one_batch(batch_size, env_id)
            avg_loss = loss_mean.item()

            grad_k.append(mean_grad_k)
            grad_b.append(mean_grad_b)

            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        if epoch == self.trainer_params['epochs'] - 1:
            grad_k = torch.mean(torch.stack(grad_k), dim=0)
            grad_b = torch.mean(torch.stack(grad_b), dim=0)
            self.model.register_buffer('grad_k', grad_k)
            self.model.register_buffer('grad_b', grad_b)
            print('last epoch, register grad', )
        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

