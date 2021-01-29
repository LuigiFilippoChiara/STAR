import torch
import torch.nn as nn

from .star import STAR
from .utils import *

from tqdm import tqdm


class processor(object):
    def __init__(self, args):

        self.args = args
        self.dataloader = Trajectory_Dataloader(args)
        self.device = self.set_device()

        self.net = STAR(args, self.device)
        self.set_optimizer()
        self.net = self.net.to(self.device)

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        self.save_model_architecture()
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
        """
        Set optimizer and loss function
        """
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.args.learning_rate)
        # MSE: Mean Squared Error
        self.criterion = nn.MSELoss(reduction='none')

    def save_model_architecture(self):
        net_file = open(
            os.path.join(self.args.model_dir, 'net.txt'), 'w')
        net_file.write(str(self.net))
        net_file.close()

    def test(self):
        """
        Load a pre-trained model and test it on the test set (1 epoch)
        """
        self.load_model()
        print('Test begun')
        self.net.eval()
        test_error, test_final_error = self.test_epoch()
        print('Set: {}, epoch: {}, test_ADE: {}, test_FDE: {}'.format(
            self.args.test_set, self.args.load_model, test_error,
            test_final_error))

    def train(self):
        """
        Train the model from scratch. Loop over the epochs, train and update
        network parameters, check results on the test set, save results and
        log data.
        """
        # TODO: what if I want to load the model and continue training?
        print('Training begun')
        test_ade, test_fde = 0, 0  # ADE, FDE

        # TODO: write header only at the beginning (check load model)
        with open(self.log_file_curve, 'w') as f:
            f.write("Epoch,Learning_rate,"
                    "Train_loss,"
                    "Train_ADE,Test_FDE,"
                    "Test_ADE,Test_FDE\n")

        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss, train_ade, train_fde = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                self.net.eval()
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
        """
        Train one epoch of the model on the whole train set.
        May print intermediate results during the epoch.
        """

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0  # Initialize epoch loss
        ade_epoch, fde_epoch = 0, 0,  # ADE, FDE
        ade_cnt, fde_cnt = 1e-5, 1e-5  # ADE, FDE denominators

        # loop over train batches
        for batch in range(self.dataloader.trainbatchnums):

            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])

            loss = torch.zeros(1).to(self.device)  # batch loss
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            # TODO: Why would I take away the last temporal step?! len=19
            # TODO: inputs are from time-step 0 to 19
            # TODO: outputs are from time-step 1 to 20
            # take away the last temporal step
            inputs_forward = batch_abs[:-1], batch_norm[:-1],\
                             shift_value[:-1], seq_list[:-1],\
                             nei_list[:-1], nei_num[:-1], batch_pednum

            self.net.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)

            loss_mask = getLossMask(outputs, seq_list[0], seq_list[1:],
                                    using_cuda=self.args.using_cuda)
            # Compute an averaged Mean-Squared-Error, only on the positions
            # in which loss_mask is True
            squared_error = self.criterion(outputs, batch_norm[1:])
            # sum Xs and Ys --> Shape becomes: seq_len*N_pedestrians
            loss_output = torch.sum(squared_error, dim=2)
            # TODO: it seems that the model is learning to compute the next
            #  step, and I am taking into account in the loss also the second
            #  step given the first step. I should account only for steps > 8
            # I divide by loss_mask.sum() instead of seq_len*N_pedestrians
            loss += (torch.sum(loss_output * loss_mask / loss_mask.sum()))
            loss_epoch += loss.item()

            # compute gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            # Use computed grafient and update parameters with an optimizer
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
                print('train: {}/{} batches, epoch = {}, train_loss = {:.5f}, '
                      'time/batch = {:.5f}'.format(
                    batch, self.dataloader.trainbatchnums,
                    epoch, loss.item(), end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        return train_loss_epoch, ade_epoch/ade_cnt, fde_epoch/fde_cnt

    @torch.no_grad()
    def test_epoch(self):
        """
        Loop over the test set once and compute ADE and FDE
        """
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0  # ADE, FDE
        # TODO: was 1e-5, 1e-5
        error_cnt_epoch, final_error_cnt_epoch = 0, 0  # ADE, FDE denominators

        # loop over test batches with progression bar
        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

            # TODO: Why do I exclude last temporal step?
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1],\
                             seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum

            # TODO: inputs are from time-step 0 to 19
            # TODO: outputs are from time-step 1 to 20
            all_output = []
            for i in range(self.args.sample_num):  # generate samples
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)
            # TODO: in theory I do not need this if I use the decorator.
            # self.net.zero_grad()

            all_output = torch.stack(all_output)

            loss_mask = getLossMask(all_output, seq_list[0], seq_list[1:],
                                    using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt = L2forTestS(
                outputs=all_output, targets=batch_norm[1:, :, :2],
                obs_length=self.args.obs_length, loss_mask=loss_mask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    def set_device(self):
        """
        Set the device for experiment. Default should be GPU.
        Set GPU if available, else CPU.
        """
        if torch.cuda.is_available() and self.args.using_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print('Using device:', device)

        # Additional info when using cuda
        if device.type == 'cuda':
            print('Name:', torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('\tAllocated:',
                  round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('\tCached:',
                  round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        print()
        return device
