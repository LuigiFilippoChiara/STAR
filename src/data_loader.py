import os
import pickle
import random

import numpy as np
from progress.bar import Bar


# Mapping dataset name to number
DATASET_NAME_TO_NUM = {
    'eth': 0,
    'hotel': 1,
    'zara1': 2,
    'zara2': 3,
    'univ': 4,
    'sdd': 5,
    'lyft': 6,
    'ind0': 7,
    'ind1': 8,
    'ind2': 9,
    'ind3': 10
}


class Trajectory_Data_Loader():
    def __init__(self, args):

        self.args = args
        self.args.test_set_name = args.test_set
        self.dataset_paths_and_dirs()

        # Trajectories files
        trajectories_path = os.path.join(
            self.args.save_dir, 'data_batches')
        if not os.path.exists(trajectories_path):
            os.makedirs(trajectories_path)

        # Train/test files
        self.train_data_file = os.path.join(trajectories_path, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(trajectories_path, "test_trajectories.cpkl")
        # Train/test batch files
        self.train_batch_file = os.path.join(trajectories_path, "train_batches.cpkl")
        self.test_batch_file = os.path.join(trajectories_path, "test_batches.cpkl")

        print("Creating pre-processed data from raw data ...")
        if not os.path.exists(self.train_data_file):
            self.read_data('train')
        if not os.path.exists(self.test_data_file):
            self.read_data('test')
        print("Data pre-processed!\n")

        # Load the processed data from the pickle files
        print("Preparing data batches ...")
        print(f'Validating on {self.args.validation_set} set')
        if not (os.path.exists(self.train_batch_file)):
            self.frameAgent_dict, self.agentTraject_dict = self.load_dict(self.train_data_file)
            self.create_data_batches('train')
        if not (os.path.exists(self.test_batch_file)):
            self.test_frameAgent_dict, self.test_agentTraject_dict = self.load_dict(self.test_data_file)
            self.create_data_batches('test')

        if self.args.validation_set == 'validation':
            self.train_batches, self.train_batches_num, self.valid_batches, \
            self.valid_batches_num = self.load_batches(
                self.train_batch_file)
            self.test_batches, self.test_batches_num = self.load_batches(
                self.test_batch_file)
        elif self.args.validation_set == 'test':
            self.train_batches, self.train_batches_num = self.load_batches(
                self.train_batch_file)
            self.valid_batches, self.valid_batches_num = self.load_batches(
                self.test_batch_file)
            self.test_batches, self.test_batches_num = self.load_batches(
                self.test_batch_file)
        else:
            raise NotImplementedError(
                f'Wrong validation set rule: {self.args.validation_set}')

        print('Total number of training batches:', self.train_batches_num)
        print('Total number of validation batches:', self.valid_batches_num)
        print('Total number of test batches:', self.test_batches_num)
        print('Data batches created and loaded!\n')

    def dataset_paths_and_dirs(self):
        """
        Define paths and directories, with respect to the chosen dataset.
        Train set, test set and skip (frame deltas between frames_id,
        per scene) are also defined here.
        """
        print("Dataset: {}, Test set: {}\n"
              .format(self.args.dataset, self.args.test_set))
        # ETH/UCY datasets
        if self.args.dataset == 'eth5':
            self.data_dirs = [
                './data/eth/univ',
                './data/eth/hotel',
                './data/ucy/zara/zara01',
                './data/ucy/zara/zara02',
                './data/ucy/univ/students001',
                './data/ucy/univ/students003',
                './data/ucy/univ/uni_examples',
                './data/ucy/zara/zara03']

            skip = [6, 10, 10, 10, 10, 10, 10, 10]

            assert self.args.test_set in ['eth', 'hotel', 'zara1', 'zara2', 'univ'], \
                'Unsupported test set {}. For eth5 dataset, test set must ' \
                'be in [eth, hotel, zara1, zara2, univ]'.format(
                    self.args.test_set)

            # set test_set, and shift it to be in [0,1,2,3,4]
            self.args.test_set = DATASET_NAME_TO_NUM[self.args.test_set] - \
                                 DATASET_NAME_TO_NUM['eth']

            if self.args.test_set == 4:
                self.test_set = [4, 5]
            else:
                self.test_set = [self.args.test_set]

            train_set = [i for i in range(len(self.data_dirs))]
            # Leave-one-scene-out validation
            for x in self.test_set:
                train_set.remove(x)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]

        # For SDD, Lyft or InD_v1 use train (70%), val (10%) and test (20%).
        # SDD dataset
        elif self.args.dataset == 'sdd':
            self.data_dirs = [
                './data/sdd/bookstore_0',
                './data/sdd/bookstore_1',
                './data/sdd/bookstore_2',
                './data/sdd/bookstore_3',
                './data/sdd/coupa_3',
                './data/sdd/deathCircle_0',
                './data/sdd/deathCircle_1',
                './data/sdd/deathCircle_2',
                './data/sdd/deathCircle_3',
                './data/sdd/deathCircle_4',
                './data/sdd/gates_0',
                './data/sdd/gates_1',
                './data/sdd/gates_3',
                './data/sdd/gates_4',
                './data/sdd/gates_5',
                './data/sdd/gates_6',
                './data/sdd/gates_7',
                './data/sdd/gates_8',
                './data/sdd/hyang_4',
                './data/sdd/hyang_5',
                './data/sdd/hyang_6',
                './data/sdd/hyang_7',
                './data/sdd/hyang_9',
                './data/sdd/nexus_0',
                './data/sdd/nexus_1',
                './data/sdd/nexus_2',
                './data/sdd/nexus_3',
                './data/sdd/nexus_4',
                './data/sdd/nexus_7',
                './data/sdd/nexus_8',
                './data/sdd/nexus_9']

            skip = [12] * len(self.data_dirs)

            assert self.args.test_set == 'sdd', \
                'When using SDD dataset, the test set must be SDD too.'

            self.train_dir = [data_dir + '_train_val/' for data_dir in self.data_dirs]
            self.test_dir = [data_dir + '_test/' for data_dir in self.data_dirs]
            self.trainskip = skip
            self.testskip = skip

        # LYFT dataset
        elif self.args.dataset == 'lyft':
            self.data_dirs = ['./data/lyft']

            skip = [1] * len(self.data_dirs)

            assert self.args.test_set == 'lyft', \
                'When using LYFT dataset, the test set must be LYFT too.'

            self.train_dir = [self.data_dirs[0] + '/lyft_train_val']
            self.test_dir = [self.data_dirs[0] + '/lyft_test']
            self.trainskip = skip
            self.testskip = skip

        # InD_v3 dataset
        elif self.args.dataset == 'ind_v3':
            self.data_dirs = [
                './data/inD_v3/' + str(i).zfill(2) for i in range(33)]

            skip = 10  # Same skip for all the recordings

            assert self.args.test_set in ['ind0', 'ind1', 'ind2', 'ind3'], \
                'When using inD dataset, test set must be in [ind0, ind1, ind2, ind3]'

            # set test_set, and shift it to be in [0,1,2,3]
            self.test_set = DATASET_NAME_TO_NUM[self.args.test_set] - \
                            DATASET_NAME_TO_NUM["ind0"]

            # Train on three scenes and test on the remaining one
            train_set = [i for i in range(4)]
            train_set.remove(self.test_set)

            # Start/stop indices of each scene
            ind_start_stop_idx = [
                [0, 6],
                [7, 17],
                [18, 29],
                [30, 32]]

            self.train_dir = [self.data_dirs[i] for idx_set in train_set
                              for i in range(ind_start_stop_idx[idx_set][0],
                                             ind_start_stop_idx[idx_set][1] + 1)]
            self.test_dir = self.data_dirs[ind_start_stop_idx[self.test_set][0]:
                                           ind_start_stop_idx[self.test_set][1] + 1]
            self.trainskip = [skip for _ in range(len(self.train_dir))]
            self.testskip = [skip for _ in range(len(self.test_dir))]
        else:
            raise NotImplementedError("Not recognized input dataset name.")

    def read_data(self, set_name):
        """
        Read raw datasets and save them into a single picke file

        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data will be stored
        """
        print(set_name.capitalize(), "set:")
        if set_name == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:  # test
            data_dirs = self.test_dir
            data_file = self.test_data_file

        # Agent IDs contained in eachframe
        frameAgent_dict = []  # frame_id: [agents_ids]
        # Trajectories of each agent
        agentTrajec_dict = []  # agent_id: trajectory

        # For each dataset
        for seti, directory in enumerate(data_dirs):

            file_path = os.path.join(directory, 'true_pos_.csv')

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Select car or pedestrian agents for InD dataset
            if self.args.dataset in ['ind_v1', 'ind_v2', 'ind_v3']:
                # data = data[:, data[-1] == self.ind_agent]
                # # Select 20% of data if not 30,31,32 (too few peds)
                if directory[-2:] not in ['30', '31', '32']:
                    data = data[:, :int(data.shape[1]*0.2)]

            # Frame IDs of the frames in the current dataset
            Pedlist = np.unique(data[1, :]).tolist()

            print(f"Processing {len(Pedlist)} agents from {directory} ...")

            # Initialize output dictionaries
            frameAgent_dict.append({})
            agentTrajec_dict.append({})

            # Progress bar
            bar = Bar('Processing agents', max=len(Pedlist))

            for ind, agent_i in enumerate(Pedlist):

                # Progress bar
                bar.next()

                # Extract trajectories of agent_i
                dataAgent = data[:, data[1, :] == agent_i]
                # Extract peds list
                FrameList = dataAgent[0, :].tolist()

                # Remove agents appearing only in one frame
                if len(FrameList) < 2:
                    continue
                # Initialize the row of the numpy array
                Trajectories = []

                # For each frame of current agent
                for frame in FrameList:
                    # Extract x and y positions
                    # TODO: ETH, HOTEL also need to be reflected on y coord???
                    if self.args.dataset == 'eth5' and os.path.basename(
                            directory) in ['univ', 'hotel']:  # Swap coordinates for ETH, HOTEL
                        current_x = dataAgent[3, dataAgent[0, :] == frame][0]
                        current_y = dataAgent[2, dataAgent[0, :] == frame][0]
                    else:
                        current_x = dataAgent[2, dataAgent[0, :] == frame][0]
                        current_y = dataAgent[3, dataAgent[0, :] == frame][0]
                    # Add frame, x and y positions to the row of the numpy array
                    Trajectories.append([int(frame), current_x, current_y])
                    # Add current frame to frameAgent_dict
                    if int(frame) not in frameAgent_dict[seti]:
                        frameAgent_dict[seti][int(frame)] = []
                    frameAgent_dict[seti][int(frame)].append(agent_i)
                agentTrajec_dict[seti][agent_i] = np.array(Trajectories)
            # Close progress bar
            bar.finish()

        with open(data_file, "wb") as f:
            # frameAgent_dict -> [dataset_index][frame] = [agent IDs in each frame]
            # agentTrajec -> [dataset_index][agentID] = [frameIDs, x, y]
            pickle.dump((frameAgent_dict, agentTrajec_dict), f, protocol=2)

    def get_data_index(self, data_dict, set_name, ifshuffle=True):
        """
        Get the dataset sampling index.
        """
        set_id = []
        frame_id_in_set = []
        total_frame = 0

        if set_name == 'train':
            skip = self.trainskip
        else:  # test
            skip = self.testskip

        for seti, dict in enumerate(data_dict):
            frames = sorted(dict)
            maxframe = max(frames) - self.args.seq_length * skip[seti]
            frames = [x for x in frames if not x > maxframe]
            total_frame += len(frames)
            set_id.extend(list(seti for i in range(len(frames))))
            frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))

        all_frame_id_list = list(i for i in range(total_frame))

        data_index = np.concatenate((
            np.array([frame_id_in_set], dtype=int),
            np.array([set_id], dtype=int),
            np.array([all_frame_id_list], dtype=int)), 0)

        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # Add one batch to data_index
        # to make full use of the data, add the beginning at the end
        # TODO: This is useless. If I delete it, nothing changes.
        #  self.args.batch_size is not used anywhere else in the code.
        #  It is not useful to add 8 frames at the end, they are not used to
        #  build any 20-frame long fragment. I do not understand this
        #  especially when shuffling. In any case, I create a batch with the
        #  remaining extra data at the end of get_batch
        # if set_name == 'train':
        #     data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def load_dict(self, data_file):
        """
        Load pre-processed raw data
        """
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)

        frameAgent_dict = raw_data[0]
        agentTraject_dict = raw_data[1]

        return frameAgent_dict, agentTraject_dict

    def load_batches(self, data_file):
        """
        Load pre-processed batches
        """
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)

        return raw_data

    def create_data_batches(self, set_name):
        """
        Create data batches for the DataLoader object
        """
        if set_name == 'train':
            shuffle = self.args.shuffle_frames_train
            if self.args.validation_set == 'validation':
                validation_fraction = self.args.validation_fraction
                assert type(validation_fraction) == float and \
                       0 < validation_fraction < 1, \
                    "args.validation_fraction must be a float between 0 and 1"
            elif self.args.validation_set == 'test':
                validation_fraction = 0
            else:
                raise NotImplementedError(
                    f'Wrong validation set: {self.args.validation_set}')
            frameAgent_dict = self.frameAgent_dict
            agentTraject_dict = self.agentTraject_dict
            batchesFile = self.train_batch_file
        else:  # set_name == test
            shuffle = self.args.shuffle_frames_test
            validation_fraction = 0  # no validation set inside test set
            frameAgent_dict = self.test_frameAgent_dict
            agentTraject_dict = self.test_agentTraject_dict
            batchesFile = self.test_batch_file

        # data_index dim : [3, N]
        # = [[frame_id_in_set, set_id, all_frame_id_list], NtotalFrames]
        data_index = self.get_data_index(
            frameAgent_dict, set_name, ifshuffle=shuffle)

        if self.args.dataset in ['eth5', 'lyft', 'ind_v1', 'ind_v2', 'ind_v3']:
            val_index = data_index[:, :int(data_index.shape[1] * validation_fraction)]
            train_index = data_index[:, int(data_index.shape[1] * validation_fraction):]
        else:  # sdd
            # TODO: Debug SSD
            val_index = np.empty((3, 0), dtype=int)
            train_index = np.empty((3, 0), dtype=int)
            for seti in np.unique(data_index[1, :]):
                data_index_seti = data_index[:, data_index[1, :] == seti]
                val_index = np.append(val_index, data_index_seti[:, :int(data_index_seti.shape[1]*validation_fraction)], axis=1)
                train_index = np.append(train_index, data_index_seti[:, int(data_index_seti.shape[1]*validation_fraction):], axis=1)

        if validation_fraction > 0:
            valbatch = self.get_batches(frameAgent_dict, agentTraject_dict,
                                        val_index, set_name, validation=True)
            trainbatch = self.get_batches(frameAgent_dict, agentTraject_dict,
                                          train_index, set_name)
            with open(batchesFile, "wb") as f:
                    pickle.dump((trainbatch, len(trainbatch), valbatch, len(valbatch)), f, protocol=2)
        else:  # no validation set here
            trainbatch = self.get_batches(frameAgent_dict, agentTraject_dict,
                                          train_index, set_name)
            with open(batchesFile, "wb") as f:
                    pickle.dump((trainbatch, len(trainbatch)), f, protocol=2)

    def get_batches(self, frameAgent_dict, agentTraject_dict, data_index,
                    set_name, validation=False):
        """
        Create data batches.
        Note: Accumulate multiple agents from different scene/frames if there
        are few agents in the current fragment.

        Parameters
        ----------
        frameAgent_dict : list
            list of dictionaries (1 per scene), linking frame to present
            pedestrians. frameAgent_dict[set_id] = {f_id: [peds_ids]}
        agentTraject_dict : list
            list of dictionaries (1 per scene), linking pedestrains to
            trajectories. agentTraject_dict[set_id] = {ped_id: [f,x,y]}
        data_index : np.array
            np.array containing frames_id and corresponding scenes
        set_name : str
            "train" or "test"
        validation : bool
            Do I want a validation set for training set. Must be False if
            set_name is 'test'

        Returns
        -------
        batch_data_total : list
            A list of batches of the form (batch_data, Batch_id, batch_traject_set_id).
            Every batch is a tuple with data and indices.
            batch_data = (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b,
            batch_pednum) is the results of massup_batch.
            Batch_id is a list of batch_id = (cur_set, cur_frame)
            batch_traject_set_id is an np.array of shape 1 * numagents_batch
            where the entries are the set_ids of the trajectories
        """
        if set_name == 'test':
            assert not validation, "There cannot be a validation set in the test set"

        # print and progress bar
        if validation:
            print(f"Processing {data_index.shape[1]} validation frames ...")
            bar = Bar(f'Processing validation frames', max=data_index.shape[1])
        else:
            print(f"Processing {data_index.shape[1]} {set_name} frames ...")
            bar = Bar(f'Processing {set_name} frames', max=data_index.shape[1])

        batch_data_total = []  # big container of all batches

        batch_data = []  # container for a batch of data
        Batch_id = []  # indices for the previous containers [(set_id, frame)]
        traject_set_id = []  # container for trajectories to set_id indices
        numagents_batch = 0  # accumulator. Number of agents in a batch

        if set_name == 'train':
            skip = self.trainskip
        else:  # else
            skip = self.testskip

        for i in range(data_index.shape[1]):
            # Progress bar
            bar.next()

            # Extract current frame and current set
            cur_frame, cur_set, _ = data_index[:, i]
            # Extract ped in current and end frames
            agents_in_start_frame = set(frameAgent_dict[cur_set][cur_frame])
            try:
                if self.args.dataset in ['eth5', 'lyft', 'ind_v1', 'ind_v2', 'ind_v3']:
                    agents_in_end_frame = set(
                        frameAgent_dict[cur_set][cur_frame + self.args.seq_length * skip[cur_set]])
                else:  # sdd
                    agents_in_end_frame = set(
                        frameAgent_dict[cur_set][cur_frame + (self.args.seq_length - 1) * skip[cur_set]])
            except:
                continue
            # pedestrians present in the current 20 frames
            present_agents = agents_in_start_frame | agents_in_end_frame
            if (agents_in_start_frame & agents_in_end_frame).__len__() == 0:
                # if the initial pedestrians all disappear after 20 frames
                continue
            traject = ()  # tuple of trajectories for this frame, reshaped
            IFfull = []  # corresponding booleans, if trajects are full
            for agent in present_agents:
                # curr_trajec [seq_length, 3], iffull [bool], ifexistslastobs [bool]
                cur_trajec, iffull, ifexistslastobs = self.find_trajectory_fragment(
                    agentTraject_dict[cur_set][agent], cur_frame,
                    self.args.seq_length, skip[cur_set])

                if np.count_nonzero(cur_trajec[:, 0]) == 0:
                    # filter out trajectories that have no data
                    continue
                if self.args.shift_last_obs:
                    if not ifexistslastobs:
                        # Just ignore trajectories if their data doesn't
                        # exist at last observed time-step (for data shift)
                        continue
                        # Remember: ifexistslastobs implicitly limit the minimum
                        # trajectory length at 8. It is even more
                        # stringent than the following explicit condition
                if np.count_nonzero(cur_trajec[:, 0]) < self.args.min_traj_length:
                    # filter out trajectories that have less than
                    # min_traj_length time-steps
                    continue

                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                # if after the conditions there are no good trajectories left
                continue
            if sum(IFfull) < 1:
                # if no trajectory is full. I want at least one per fragment
                continue

            # traject_batch.shape: seq_len*N_pedestrians*(x,y)
            traject_batch = np.concatenate(traject, axis=1)
            cur_agentsnum = traject_batch.shape[1]
            # Append(set_id of each trajectory)
            # Append cur_set value times number of agents
            traject_set_id.append(np.full(
                (1, cur_agentsnum), fill_value=cur_set))
            numagents_batch += cur_agentsnum
            batch_id = (cur_set, cur_frame)

            # accumulate multiple frame_data into a batch
            batch_data.append(traject_batch)
            Batch_id.append(batch_id)
            # enough people in the batch
            if numagents_batch > self.args.batch_around_agent:
                batch_data = self.massup_batch(batch_data)
                # batch_traject_set_id.shape : 1 * numagents_batch
                batch_traject_set_id = np.hstack(traject_set_id)
                batch_data_total.append((batch_data, Batch_id, batch_traject_set_id))

                batch_data = []
                Batch_id = []
                traject_set_id = []
                numagents_batch = 0
        bar.finish()  # Close progress bar

        # if there are remaining pedestrians left outside, create a final batch.
        # I need all the pedestrians for the test set.
        # For the training set, I prefer not to have an incomplete batch.
        if numagents_batch > 0 and set_name == 'test':
            batch_data = self.massup_batch(batch_data)
            batch_traject_set_id = np.hstack(traject_set_id)
            batch_data_total.append((batch_data, Batch_id, batch_traject_set_id))

        return batch_data_total

    def find_trajectory_fragment(self, trajectory, start_frame, seq_length, skip):
        """
        Starting from a full trajectory, query a trajectory fragment of
        length seq_length starting at start_frame. Replace with 0 if data
        does not exists.

        Parameters
        ----------
        trajectory : np.array
            full trajectory of an agent. Shape: len_traj * [frame,x,y]
            len_traj really depends on the considered agent, it can be
            shorter or longer than seq_length.
        start_frame : int
            Starting frame of current trajectory fragment
        seq_length : int
            Desired length of the trajectory fragment
        skip : int
            Frame_delta between two time-steps

        Returns
        -------
        return_trajec : np.array
            The trajectory fragment of shape seq_length * (x,y)
        iffull : bool
            True if the trajectory fragment is full (seq_length full)
        ifexsitslastobs : bool
            True if return_trajec has data at the last observation time-step
        """
        return_trajec = np.zeros((seq_length, 3))
        iffull = False
        ifexsitslastobs = False
        start_n = np.where(trajectory[:, 0] == start_frame)

        if self.args.dataset in ['eth5', 'lyft', 'ind_v1', 'ind_v2', 'ind_v3']:
            end_frame = start_frame + seq_length * skip
            end_n = np.where(trajectory[:, 0] == end_frame)
        else:  # sdd
            end_frame = start_frame + (seq_length - 1) * skip
            end_n = np.where(trajectory[:, 0] == end_frame) + np.array([1])

        # if no start and end
        if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
            start_n = 0
            end_n = end_n[0][0]
            # trajectory ends at start_frame. Return zeros
            if end_n == 0:
                return return_trajec, iffull, ifexsitslastobs
        # if start and no end
        elif start_n[0].shape[0] != 0 and end_n[0].shape[0] == 0:
            start_n = start_n[0][0]
            end_n = trajectory.shape[0]
        # if no start and no end
        elif start_n[0].shape[0] == 0 and end_n[0].shape[0] == 0:
            # if start_frame and end_frame included in trajectory
            if start_frame < trajectory[0, 0] and end_frame > trajectory[-1, 0]:
                start_n = 0
                end_n = trajectory.shape[0]
            else:
                # if start_frame and end_frame completely out of trajectory
                # This never happens. Return zeros in case
                return return_trajec, iffull, ifexsitslastobs
        # if start and end
        else:
            end_n = end_n[0][0]
            start_n = start_n[0][0]

        candidate_seq = trajectory[start_n:end_n]

        # number of zeros at the start of return_trajec
        offset_start = int((candidate_seq[0, 0] - start_frame) // skip)

        if self.args.dataset in ['eth5', 'lyft', 'ind_v1', 'ind_v2', 'ind_v3']:
            offset_end = self.args.seq_length + int(
                (candidate_seq[-1, 0] - end_frame)//skip)
        # TODO: try with sdd
        else:  # sdd
            offset_end = self.args.seq_length + int(
                (candidate_seq[-1, 0] - end_frame)//skip) - 1

        # place the candidate_seq at the right offset
        return_trajec[offset_start:offset_end + 1, :3] = candidate_seq

        if return_trajec[self.args.obs_length - 1, 1] != 0:
            # if return_trajec has data at the last observation time-step
            ifexsitslastobs = True
        if offset_end - offset_start >= seq_length - 1:
            # you need to do this. Frame, x or y may be zeros (but valid data)
            iffull = True

        return return_trajec, iffull, ifexsitslastobs

    def massup_batch(self, batch_data):
        """
        Mass up data fragments in different time windows together to a batch.
        Aggregate a list of arrays trajectory fragments into 4 big arrays.
        nodes_batch_batch contains the aggregated trajectory data,
        while seq_list_batch, nei_list_batch, nei_num_batch contains social
        information.

        Parameters
        ----------
        batch_data : list
            list of np.arrays of shape seq_length * N_pedestrians * (x,y)
            Each array corresponds to a different time window/scene.
            N_pedestrians usually differ for each array of the list.

        Returns
        -------
        nodes_batch_batch : np.array
            trajectory data, aggregated over different time fragments.
            Shape: seq_length * num_Peds * (x,y)
        seq_list_batch : np.array
            boolean index, True when trajectory data exists. Shape:
            seq_length*num_Peds
        nei_list_batch : np.array
            boolean index, nei_list_b[f,i,j] is True when i is j's neighbor
            at time-step f. Shape: seq_length*num_Peds*num_Peds
        nei_num_batch : np.array
            neighbors count for each pedestrian. Shape: seq_length * num_Peds
        batch_agentnum : list
            list of pedestrian number in the same fragment, as in the input
            batch_data.
        """

        num_Agents = 0  # number of pedestrians in batch_data
        for batch in batch_data:
            num_Agents += batch.shape[1]

        seq_list_batch = np.zeros((self.args.seq_length, 0))
        nodes_batch_batch = np.zeros((self.args.seq_length, 0, 2))
        nei_list_batch = np.zeros((self.args.seq_length, num_Agents, num_Agents))
        nei_num_batch = np.zeros((self.args.seq_length, num_Agents))
        agent_pointer = 0  # pedestrian number accumulator inside for loop
        batch_agentnum = []  # pedestrian numbers per fragment in the batch

        for batch in batch_data:
            num_Agent = batch.shape[1]
            seq_list, nei_list, nei_num = self.get_neighbourhood(
                batch, self.args.neighbor_shape)
            nodes_batch_batch = np.append(nodes_batch_batch, batch, axis=1)
            seq_list_batch = np.append(seq_list_batch, seq_list, axis=1)
            nei_list_batch[:, agent_pointer:agent_pointer + num_Agent, agent_pointer:agent_pointer + num_Agent] = nei_list
            nei_num_batch[:, agent_pointer:agent_pointer + num_Agent] = nei_num
            batch_agentnum.append(num_Agent)
            agent_pointer += num_Agent

        return nodes_batch_batch, seq_list_batch, nei_list_batch, nei_num_batch, batch_agentnum

    def get_neighbourhood(self, batch, neighbor_shape='circle'):
        """
        Define the social neighbourhood structure for a trajectory fragment.
        Get the sequence list (denoting where data exists), neighbourhoods
        list (denoting where neighbors exists) and neighbors counts.

        Parameters
        ----------
        batch : np.array
            trajectory fragment. Shape: seq_len(20) * N_agents * (x,y)
        neighbor_shape : str
            Shape of the neighbourhood. Can be circle (2 norm) or square

        Returns
        -------
        seq_list : np.array
            Boolean mask of shape seq_len(20) * N_agents.
            seq_list[f,i]=1 if batch[f,i] data exists
        nei_list : np.array
            Boolean mask of shape seq_len(20) * N_agents * N_agents.
            nei_list[f,i,j] denote if j is i's neighbor in frame f
        nei_num : np.array
            Integer count of shape seq_len(20) * N_agents.
            nei_list[f,i] denotes the number of neighbors of agent i in frame f
        """
        num_Agents = batch.shape[1]

        assert neighbor_shape in ['circle', 'square']

        # denote where data not missing
        seq_list = np.zeros((batch.shape[0], num_Agents))
        for pedi in range(num_Agents):
            seq = batch[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # neighbourhood id list. Default is no neighbors
        # nei_list[f,i,j] denote if j is i's neighbor in frame f
        nei_list = np.zeros((batch.shape[0], num_Agents, num_Agents))

        for pedi in range(num_Agents):
            seqi = batch[:, pedi]
            for pedj in range(num_Agents):
                # person i is not neighbor of itself
                if pedi == pedj:
                    continue
                both_present = seq_list[:, [pedi, pedj]].all(axis=1)
                seqj = batch[:, pedj]
                relative_cord = seqi[:, :2] - seqj[:, :2]
                # select_dist: indices where pedi and pedj are neighbors
                if neighbor_shape == 'circle':
                    # inside a circle of radius self.args.neighbor_thred
                    select_dist = np.linalg.norm(relative_cord, ord=2, axis=1) < self.args.neighbor_thred
                else:  # square
                    # inside a square of side self.args.neighbor_thred
                    select_dist = np.all(np.abs(relative_cord) < self.args.neighbor_thred, axis=1)
                neighbors_bool = np.logical_and(both_present, select_dist)
                nei_list[:, pedi, pedj] = neighbors_bool

        # number of neighbors
        nei_num = nei_list.sum(axis=1)

        # TODO: get rid of these tests once merged
        # TESTS
        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)
        def check_all_symmetric(arr):
            return all(check_symmetric(a) for a in arr)
        assert nei_num.shape == (batch.shape[0], num_Agents)
        assert check_all_symmetric(nei_list)

        return seq_list, nei_list, nei_num

    def rotate_shift_batch(self, batch_data, random_rotate=False, shift=False):
        """
        Random ration (0-180 degrees) and shifting wrt last observation.
        """
        batch, seq_list, nei_list, nei_num, batch_agentnum = batch_data

        # rotate batch randomly
        if random_rotate:
            # TODO: this rotation should be normally distributed not uniform!
            theta = random.random() * np.pi
            batch_orig = batch.copy()
            # rotation around (0,0). Note that there are fragments of
            # different scenes in a batch and they all rotate.
            batch[:, :, 0] = batch_orig[:, :, 0] * np.cos(theta) - batch_orig[:, :, 1] * np.sin(theta)
            batch[:, :, 1] = batch_orig[:, :, 0] * np.sin(theta) + batch_orig[:, :, 1] * np.cos(theta)

        # TODO: check if shift change relative positions of pedestrians!
        # shift trajectories based on last observation coordinates
        if shift:
            s = batch[self.args.obs_length - 1]
            shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)
        else:  # zero shift
            shift_value = np.zeros_like(batch)

        batch_data = batch, batch - shift_value, shift_value, seq_list, nei_list, nei_num, batch_agentnum

        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id, traject_set_id = self.train_batches[idx]
        batch_data = self.rotate_shift_batch(
            batch_data, random_rotate=self.args.random_rotate,
            shift=self.args.shift_last_obs)
        return batch_data, batch_id, traject_set_id

    def get_test_batch(self, idx):
        batch_data, batch_id, traject_set_id = self.test_batches[idx]
        batch_data = self.rotate_shift_batch(
            batch_data, random_rotate=False, shift=self.args.shift_last_obs)
        return batch_data, batch_id, traject_set_id

    def get_valid_batch(self, idx):
        batch_data, batch_id, traject_set_id = self.valid_batches[idx]
        batch_data = self.rotate_shift_batch(
            batch_data, random_rotate=False, shift=self.args.shift_last_obs)
        return batch_data, batch_id, traject_set_id
