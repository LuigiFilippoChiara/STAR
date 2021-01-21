import os
import pickle
import random
import time
import numpy as np
import torch


DATASET_NAME_TO_NUM = {
    'eth': 0,
    'hotel': 1,
    'zara1': 2,
    'zara2': 3,
    'univ': 4
}


class Trajectory_Dataloader():
    def __init__(self, args):

        self.args = args
        if self.args.dataset == 'eth5':

            self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                              './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                              './data/ucy/univ/students001', './data/ucy/univ/students003',
                              './data/ucy/univ/uni_examples', './data/ucy/zara/zara03']

            # Data directory where the pre-processed pickle file resides
            self.data_dir = './data'

            # frame deltas between frames_id, for each scene
            skip = [6, 10, 10, 10, 10, 10, 10, 10]

            train_set = [i for i in range(len(self.data_dirs))]

            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)
            print("Test set is: {} --> index = {}\n"
                  .format(args.test_set, DATASET_NAME_TO_NUM[args.test_set]))

            args.test_set = DATASET_NAME_TO_NUM[args.test_set]

            if args.test_set == 4 or args.test_set == 5:
                self.test_set = [4, 5]
            else:
                self.test_set = [self.args.test_set]

            for x in self.test_set:
                train_set.remove(x)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]
        else:
            raise NotImplementedError

        self.train_data_file = os.path.join(self.args.save_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.save_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.save_dir, "train_batch_cache.cpkl")
        self.test_batch_cache = os.path.join(self.args.save_dir, "test_batch_cache.cpkl")

        print("Creating pre-processed data from raw data ...")
        if not os.path.exists(self.train_data_file):
            self.traject_preprocess('train')
        if not os.path.exists(self.test_data_file):
            self.traject_preprocess('test')
        print("Done.\n")

        # Load the processed data from the pickle file
        print("Preparing data batches ...")
        if not (os.path.exists(self.train_batch_cache)):
            self.frameped_dict, self.pedtraject_dict = self.load_dict(self.train_data_file)
            self.dataPreprocess('train')
        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict = self.load_dict(self.test_data_file)
            self.dataPreprocess('test')

        self.trainbatch, self.trainbatchnums, _, _ = self.load_cache(self.train_batch_cache)
        self.testbatch, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache)

        print('Total number of training batches:', self.trainbatchnums)
        print('Total number of test batches:', self.testbatchnums)
        print("Done.\n")

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)

    def traject_preprocess(self, setname):
        '''
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:
            data_dirs = self.test_dir
            data_file = self.test_data_file
        all_frame_data = []
        valid_frame_data = []
        numFrame_data = []

        Pedlist_data = []
        frameped_dict = []  # frame_id: [pedestrains_ids]
        pedtrajec_dict = []  # ped_id: trajectory
        # For each dataset
        for seti, directory in enumerate(data_dirs):

            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset

            Pedlist = np.unique(data[1, :]).tolist()
            numPeds = len(Pedlist)
            # Add the list of frameIDs to the frameList_data
            Pedlist_data.append(Pedlist)
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])
            numFrame_data.append([])
            frameped_dict.append({})
            pedtrajec_dict.append({})

            # iterate over pedestrians of the current scene
            for ind, pedi in enumerate(Pedlist):
                if ind % 100 == 0:
                    print("Scene {} of {}, preprocessed pedestrians: {}/{}".format(
                        seti, setname, ind, len(Pedlist)))
                # Frame that contains that pedestrian
                FrameContainPed = data[:, data[1, :] == pedi]
                # To list
                FrameList = FrameContainPed[0, :].tolist()
                if len(FrameList) < 2:
                    # ignore the person if present in less than 2 frames
                    continue
                # Add number of frames with data
                numFrame_data[seti].append(len(FrameList))
                # Initialize the numpy trajectory array
                Trajectories = []

                # For each frame
                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                    current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                    # append pedID, x, y row to the end of the trajectory
                    Trajectories.append([int(frame), current_x, current_y])
                    if int(frame) not in frameped_dict[seti]:
                        frameped_dict[seti][int(frame)] = []
                    # append frame: [peds] to dictiornary
                    frameped_dict[seti][int(frame)].append(pedi)
                # append ped: Traj to dictorionary
                pedtrajec_dict[seti][pedi] = np.array(Trajectories)

        f = open(data_file, "wb")
        pickle.dump((frameped_dict, pedtrajec_dict), f, protocol=2)
        f.close()

    def get_data_index(self, data_dict, setname, ifshuffle=True):
        '''
        Get the dataset sampling index.
        '''
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        for seti, dict in enumerate(data_dict):
            frames = sorted(dict)
            # TODO Why are we subtracting 20 frames and not 20 timesteps?!
            #?! maxframe = max(frames) - self.args.seq_length*(frames[1] -
            # frames[0])
            maxframe = max(frames) - self.args.seq_length
            frames = [x for x in frames if not x > maxframe]
            total_frame += len(frames)
            set_id.extend(list(seti for i in range(len(frames))))
            frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))

        all_frame_id_list = list(i for i in range(total_frame))

        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # to make full use of the data, add the beginning at the end
        # TODO: Why do I need this?! self.args.batch_size is really
        #  useful?! Shouldn't self.args.obs_length or self.args.batch_around_ped
        #  be used instead?
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]

        return frameped_dict, pedtraject_dict

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        print("{} set:".format(setname.capitalize()))

        if setname == 'train':
            val_fraction = 0
            frameped_dict = self.frameped_dict
            pedtraject_dict = self.pedtraject_dict
            cachefile = self.train_batch_cache

        else:
            val_fraction = 0
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            cachefile = self.test_batch_cache

        if setname != 'train':
            shuffle = False
        else:
            shuffle = True

        data_index = self.get_data_index(frameped_dict, setname, ifshuffle=shuffle)
        val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]
        trainbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, train_index, setname)
        valbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, val_index, setname)
        trainbatchnums = len(trainbatch)
        valbatchnums = len(valbatch)

        f = open(cachefile, "wb")
        pickle.dump((trainbatch, trainbatchnums, valbatch, valbatchnums), f, protocol=2)
        f.close()

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname):
        """
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the
        scene if there are few people. This function takes less gpu memory.

        Parameters
        ----------
        frameped_dict : list
            list of dictionaries (1 per scene), linking frame to present
            pedestrians
        pedtraject_dict : list
            list of dictionaries (1 per scene), linking pedestrain to
            trajectories
        data_index : np.array
            np.array containing frames_id and corresponding scenes
        setname : str
            "train" or "test"

        Returns
        -------
        list
            batch_data_mass: a list of batches of the form (batch_data,
            Batch_id). Every batch is a tuple with data and indices.
            batch_data = (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b,
            batch_pednum) is the results of massup_batch.
            Batch_id is a list of batch_id = (cur_set, cur_frame)
        """
        batch_data_mass = []  # big container of all batches

        batch_data = []  # container for a batch of data
        Batch_id = []  # indices for the previous containers

        if setname == 'train':
            skip = self.trainskip
        else:
            skip = self.testskip

        ped_cnt = 0  # accumulator of all analysed pedestrains
        last_frame = 0

        # loop over frames
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print("Processed frames:", i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            # pedestrians in cur_frame
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            try:
                # pedestrians in the end frame (+20 frames)
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + self.args.seq_length * skip[cur_set]])
            except:
                continue
            # pedestrians present in the current 20 frames
            present_pedi = framestart_pedi | frameend_pedi
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                # if the initial pedestrians all disappear after 20 frames
                continue
            traject = ()  # tuple of trajectories for this frame, reshaped
            IFfull = []  # corresponding booleans, if trajects are full
            for ped in present_pedi:
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped], cur_frame,
                                                                               self.args.seq_length, skip[cur_set])
                if len(cur_trajec) == 0:
                    continue
                if not ifexistobs:
                    # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # TODO: why 5?
                    # filter out trajectories that have less than 5 time-steps
                    continue

                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                # I want at least one full traj per frame
                continue
            # traject_batch.shape: seq_len*N_pedestrians*2
            traject_batch = np.concatenate(traject, axis=1)
            # number of ped in current fragmnet
            cur_pednum = traject_batch.shape[1]
            # prosessive number of pedestrians in the batch
            batch_pednum = sum([i.shape[1] for i in batch_data]) + cur_pednum

            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame)

            if cur_pednum >= self.args.batch_around_ped * 2:
                assert cur_pednum < self.args.batch_around_ped * 2, \
                    "too many people ({}) in current fragment. Better to " \
                    "increase batch_around_ped (current: {}) !" \
                    "".format(cur_pednum, self.args.batch_around_ped)
                # too many people in current fragment
                # --> split the fragment into two batches
                # indices to sort the pedestrians in the fragment
                ind = traject_batch[self.args.obs_length - 1].argsort(axis=0)
                # pedestrian are splitted in 2, based on the x position
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]], traject_batch[:, ind[cur_pednum // 2:, 0]]]
                for sb in Seq_batchs:
                    cur_batch_data, cur_Batch_id = [], []
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_batch_data = self.massup_batch(cur_batch_data)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id))
                last_frame = i

            elif cur_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                # create one ad-hoc batch for this fragment only
                cur_batch_data, cur_Batch_id = [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_batch_data = self.massup_batch(cur_batch_data)
                batch_data_mass.append((cur_batch_data, cur_Batch_id))
                last_frame = i

            else:  # cur_pednum < self.args.batch_around_ped (most common)
                # --> accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped:
                    # enough people have been accumulated
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)

                    batch_data = self.massup_batch(batch_data)
                    batch_data_mass.append((batch_data, Batch_id))

                    last_frame = i
                    # reinitialise batch data
                    batch_data = []
                    Batch_id = []
                else:
                    # continue to accumulate people (most common)
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)

        # if there are remaining pedestrians left outside
        # TODO: Why only test?
        if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            batch_data_mass.append((batch_data, Batch_id))
        return batch_data_mass

    def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip):
        """
        Starting from a full trajectory, query the trajectory fragment
        starting at startframe. Replace with 0 if data does not exist.
        """
        return_trajec = np.zeros((seq_length, 3))
        endframe = startframe + seq_length * skip
        start_n = np.where(trajectory[:, 0] == startframe)
        end_n = np.where(trajectory[:, 0] == endframe)
        iffull = False
        ifexsitobs = False

        if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
            start_n = 0
            end_n = end_n[0][0]
            if end_n == 0:
                return return_trajec, iffull, ifexsitobs

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
            start_n = start_n[0][0]
            end_n = trajectory.shape[0]

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
            start_n = 0
            end_n = trajectory.shape[0]

        else:
            end_n = end_n[0][0]
            start_n = start_n[0][0]

        candidate_seq = trajectory[start_n:end_n]
        offset_start = int((candidate_seq[0, 0] - startframe) // skip)

        offset_end = self.args.seq_length + int((candidate_seq[-1, 0] - endframe) // skip)

        return_trajec[offset_start:offset_end + 1, :3] = candidate_seq

        if return_trajec[self.args.obs_length - 1, 1] != 0:
            ifexsitobs = True

        if offset_end - offset_start >= seq_length - 1:
            iffull = True

        return return_trajec, iffull, ifexsitobs

    def massup_batch(self, batch_data):
        """
        Mass up data fragments in different time windows together to a batch.
        Aggregate a list of arrays trajectory fragments into 4 big arrays.
        nodes_batch_b contains the aggregated trajectory data,
        while seq_list_b, nei_list_b, nei_num_b contains social information.

        Parameters
        ----------
        batch_data : list
            list of np.arrays of shape seq_length*N_pedestrians*(x,y)
            Each array corresponds to a different time window/scene.
            N_pedestrians may differ for each array of the list.

        Returns
        -------
        tuple
            (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)
            nodes_batch_b: trajectory data, aggregated over different time
            fragments. Shape: seq_length*num_Peds*(x,y)
            seq_list_b: boolean index, True when trajectory data exists. Shape:
            seq_length*num_Peds
            nei_list_b: boolean index, nei_list_b[f,i,j] is True when when i
            is j's neighbor at time-step f. Shape: seq_length*num_Peds*num_Peds
            nei_num_b: neighbors count for each pedestrian. Shape: seq_length*num_Peds
            batch_pednum: list of pedestrian number in the same fragment,
            as in the input batch_data.
        """
        num_Peds = 0 # number of pedestrians in batch_data
        for batch in batch_data:
            num_Peds += batch.shape[1]

        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 2))
        nei_list_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        nei_num_b = np.zeros((self.args.seq_length, num_Peds))
        num_Ped_h = 0  # pedestrian number accumulator inside for loop
        batch_pednum = []  # pedestrian numbers per fragment in the batch
        for batch in batch_data:
            num_Ped = batch.shape[1]
            seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch)
            nodes_batch_b = np.append(nodes_batch_b, batch, axis=1)
            seq_list_b = np.append(seq_list_b, seq_list, axis=1)
            nei_list_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            nei_num_b[:, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        return nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum

    # TODO: redo this function (have a loke at the notebook)
    def get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exists) and neighboring
        list (denoting where neighbors exists).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            # the default is that everyone is neighbor (where data exists)
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], axis=1)  # count
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                seqj = inputnodes[:, pedj]
                # where both sequences have data
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                relative_cord = seqi[select, :2] - seqj[select, :2]

                # indices where pedi and pedj are not neighors
                # select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                #         abs(relative_cord[:, 1]) > self.args.neighbor_thred)

                # TODO: use l2_norm > L for distance instead of x>L or y>L
                select_dist = np.linalg.norm(relative_cord,
                                               axis=1) > self.args.neighbor_thred

                nei_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0

        return seq_list, nei_list, nei_num

    def rotate_shift_batch(self, batch_data, ifrotate=True):
        """
        Random ration and zero shifting of trajectories when batches are
        loaded during training.
        """
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data

        # rotate batch
        if ifrotate:
            theta = random.random() * np.pi  # random angle between 0 and pi
            cur_ori = batch.copy()
            # rotation around (0,0). Note that there are fragments of
            # different scenes in a batch and they all rotate.
            batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(theta) - cur_ori[:, :, 1] * np.sin(theta)
            batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(theta) + cur_ori[:, :, 1] * np.cos(theta)
        # get shift value (positions at the end of the observation time)
        s = batch[self.args.obs_length - 1]

        shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)

        batch_data = batch, batch - shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id = self.trainbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)

        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.testbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False)
        return batch_data, batch_id

    def reset_batch_pointer(self, set, valid=False):
        '''
        Reset all pointers
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0


def getLossMask(outputs, node_first, seq_list, using_cuda=False):
    """
    Get a mask to denote whether both of current and previous data exist.
    Note: It is not supposed to calculate loss for a person at time t if his
    data at t-1 does not exist.

    Parameters
    ----------
    outputs : PyTorch tensor
        Output of the prediction model.
        If stochastic size is: num_sample*seq_len*N_pedestrians*(x,y)
        If deterministic size is: seq_len*N_pedestrians*(x,y)
    node_first : PyTorch tensor
        input is seq_list[0]. Size = N_pedestrians. Boolean mask that is =1
        if pedestrian i is present at time-step 0.
    seq_list : PyTorch tensor
        input is seq_list[1:]. Size = (seq_len-1)*N_pedestrians. Boolean mask
        that is =1 if pedestrian i is present at time-step t >= 1.

    Returns
    -------
    loss_mask : PyTorch tensor
        Shape: seq_len*N_pedestrians
        loss_mask[t,i] = 1 if pedestrian i if present at both t and t-1
    """
    if outputs.dim() == 3:  # train or deterministic test
        seq_length = outputs.shape[0]
    elif outputs.dim() == 4:  # stochastic test
        seq_length = outputs.shape[1]

    node_pre = node_first
    loss_mask = torch.zeros(seq_length, seq_list.shape[1])

    if using_cuda:
        loss_mask = loss_mask.cuda()

    # For loss mask, only generate for those exist through the whole window
    for frame_num in range(seq_length):
        if frame_num == 0:
            loss_mask[frame_num] = seq_list[frame_num] * node_pre
        else:
            loss_mask[frame_num] = seq_list[frame_num] * loss_mask[frame_num - 1]

    return loss_mask


def L2forTest(outputs, targets, obs_length, lossMask):
    """
    Evaluation function for deterministic output.
    """

    # outputs = outputs[0, :, :, :]

    assert outputs.dim() == 3, "seq_len * N_pedestrians * (x,y)"
    assert outputs.size()[0] == 19 # TODO: Why not 20?!
    assert outputs.size()[3] == 2

    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    # TODO: here again we are considering one less temporal step. Why?!
    error_full = error[obs_length - 1:, pedi_full]

    error = torch.sum(error_full)  # ADE
    error_cnt = error_full.numel()  # Denominator ADE
    final_error = torch.sum(error_full[-1])  # FDE
    final_error_cnt = error_full[-1].numel()  # Denominator for FDE

    return error.item(), error_cnt, final_error.item(), final_error_cnt


def L2forTestS(outputs, targets, obs_length, lossMask, num_samples=20):
    """
    Evaluation, stochastic version
    """

    # TODO: check if it works also for num_samples = 1 (deterministic)


    assert outputs.dim() == 4, "sample_num * seq_len * N_pedestrians * (x,y)"
    assert outputs.shape[0] == num_samples
    assert outputs.shape[1] == 19  # TODO: Why not 20?!
    assert outputs.shape[3] == 2

    assert targets.dim() == 3, "seq_len * N_pedestrians * (x,y)"
    assert outputs.shape[1:] == targets.shape

    assert obs_length == 8, "Observation length"

    assert lossMask.shape == outputs.shape[1:3]

    seq_length = outputs.shape[1]
    # compute L2 error of (x, y) distances point-wise
    # error.size() : sample_num * seq_len * N_pedestrians
    error = torch.norm(outputs - targets, p=2, dim=3)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    # TODO: here again one temporal step is forgotten..
    error_full = error[:, obs_length - 1:, pedi_full]

    # sum over time-steps
    error_full_sum = torch.sum(error_full, dim=1)
    # min over samples
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    best_error = []
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)
    # best_error.size() : pred_length (12) * N_pedestrians_full

    error = torch.sum(best_error)  # ADE
    error_cnt = best_error.numel()  # ADE denominator count

    final_error = torch.sum(best_error[-1])  # FDE
    final_error_cnt = best_error.shape[-1]  # FDE denominator count

    return error.item(), error_cnt, final_error.item(), final_error_cnt


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:',
              round((te - ts) * 1000, 1), 'ms\n')
        return result
    return timed


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
