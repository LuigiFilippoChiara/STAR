import torch
import torch.nn as nn

from .star import STAR
from .data_loader import *

from tqdm import tqdm


class processor(object):
    def __init__(self, args):

        self.args = args

        self.dataloader = Trajectory_Dataloader(args)
        self.net = STAR(args)

        self.set_optimizer()

        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()

        self.log_file_curve = os.path.join(self.args.model_dir, 'log_curve.txt')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):
        """
        Save model and optimizer states at each epoch
        """
        model_path = os.path.join(
                self.args.model_dir,
                self.args.train_model + '_' + str(epoch) + '.tar')
        torch.save({'epoch': epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   model_path)

    def load_model(self):
        """
        Load a model. Can then be used to test
        """
        # TODO: what if I want to load the model and continue training?
        if self.args.load_model is not None:
            self.args.model_save_path = os.path.join(
                self.args.model_dir,
                self.args.train_model + '_' + str(self.args.load_model) +
                '.tar')
            print("Saved model path:", self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint ...')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch, '\n')
        else:
            raise ValueError('You need to specify an epoch if you want to '
                             'load a model! Change args.load_model')

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def test(self):
        """
        Load a pre-trained model and test it on the test set (1 epoch)
        """
        self.load_model()
        print('Test begun')
        test_error, test_final_error = self.test_epoch()
        print('Set: {}, epoch: {}, test_ADE: {}, test_FDE: {}'.format(
            self.args.test_set, self.args.load_model, test_error,
            test_final_error))

    def train(self):

        print('Training begin')
        test_ade, test_fde = 0, 0  # ADE, FDE

        # TODO: write header only at the beginning (check load model)
        with open(self.log_file_curve, 'w') as f:
            f.write("Epoch,Learning_rate,"
                    "Train_loss,"
                    "Train_ADE,Train_FDE,"
                    "Test_ADE,Test_FDE\n")

        for epoch in range(self.args.num_epochs):
            train_loss, train_ade, train_fde = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                test_ade, test_fde = self.test_epoch()  # ADE, FDE

                if test_fde < self.best_fde:  # update if better FDE
                    self.best_ade = test_ade
                    self.best_fde = test_fde
                    self.best_epoch = epoch
                self.save_model(epoch)

                print('----Epoch {}, train_loss={:.5f}, '
                      'train_ADE={:.3f}, train_FDE={:.3f}, '
                      'test_ADE={:.3f}, test_FDE={:.3f}, '
                      'best_ADE={:.3f}, best_FDE={:.3f} '
                      'at epoch {}'.format(
                    epoch, train_loss,
                    train_ade, train_fde,
                    test_ade, test_fde,
                    self.best_ade, self.best_fde,
                    self.best_epoch))
            else:
                print('----epoch {}, train_loss={:.5f}'.format(
                    epoch, train_loss))

            with open(self.log_file_curve, 'a') as f:
                f.write(','.join(str(m) for m in [
                    epoch, self.args.learning_rate,
                    train_loss,
                    train_ade, train_fde,
                    test_ade, test_fde
                ]) + '\n')

    def train_epoch(self, epoch):
        self.net.train()
        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0  # Initialize epoch loss
        ade_epoch, fde_epoch = 0, 0,  # ADE, FDE
        ade_cnt, fde_cnt = 1e-5, 1e-5  # ADE, FDE denominators

        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)

            loss_mask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)

            loss += (torch.sum(loss_o * loss_mask / num))
            loss_epoch += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)

            self.optimizer.step()

            with torch.no_grad():
                error, error_cnt, final_error, final_error_cnt = L2forTestS(
                    outputs=torch.stack([outputs]),
                    targets=batch_norm[1:],
                    loss_mask=loss_mask, obs_length=self.args.obs_length)

            # used to print and log
            ade_epoch += error
            ade_cnt += error_cnt
            fde_epoch += final_error
            fde_cnt += final_error_cnt

            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss.item(),
                                                                                               end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch, ade_epoch/ade_cnt, fde_epoch/fde_cnt

    @torch.no_grad()
    def test_epoch(self):
        self.net.eval()
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5

        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[
                                                                                                              :-1], batch_pednum

            all_output = []
            for i in range(self.args.sample_num):
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)

            loss_mask, _ = getLossMask(all_output, seq_list[0], seq_list[1:],
                                       using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt = L2forTestS(
                outputs=all_output,
                targets=batch_norm[1:],
                loss_mask=loss_mask, obs_length=self.args.obs_length)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch
