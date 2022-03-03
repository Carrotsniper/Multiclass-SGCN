import math
import os

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import config

def loc_pos(seq_):

    # seq_ [obs_len N 2]

    obs_len = seq_.shape[0]
    num_ped = seq_.shape[1]

    pos_seq = np.arange(1, obs_len + 1)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat(num_ped, axis=1)

    result = np.concatenate((pos_seq, seq_), axis=-1)

    return result


def seq_to_graph(seq_, seq_rel, pos_enc=False):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))        
    # A = np.zeros((seq_len, max_nodes, max_nodes)) 
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
    
    if pos_enc:
        V = loc_pos(V)

    return torch.from_numpy(V).type(torch.float) # torch.from_numpy(A).type(torch.float)


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def read_file(_path, delim='space'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if (len(line) == 5):
                for i in range(len(line)):
                    try:
                        line[i] = float(line[i])
                    except ValueError:
                        line[i] = str(line[i])
                data.append(line)
    return np.asarray(data, dtype=object)




class AgentTrajectoryDataset(Dataset):
    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_agent=1, delim='space', norm_lap_matr=True):
        super(AgentTrajectoryDataset, self).__init__()
        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_agents_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_list_class = []
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            data = read_file(path, delim)
            if (np.array_equal(data, [])):
                print(str(path) + " - No data in file")
                continue

            frames = np.unique(data[:, 0]).tolist()
            frame_data =[]
            for frame in frames:
                frame_data.append(data[frame == data[:,0]])

            num_sequences = int(np.ceil(len(frames) - self.seq_len +1) /skip)   
            
            for idx in range(0, num_sequences * self.skip +1 ,skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx+self.seq_len], axis=0)
                
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1]) #30
                self.max_agents_in_frame = max(self.max_agents_in_frame,len(agents_in_curr_seq))

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # [30x2x20]
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len))
                curr_seq_class = np.empty((len(agents_in_curr_seq)), dtype=object)
                curr_loss_mask = np.zeros((len(agents_in_curr_seq), self.seq_len)) # [55, 20]
                num_agents_considered = 0
                _non_linear_ped = []
                
                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:,1]== agent_id, :]
                    curr_agent_seq[:, :-1] = np.round(np.asarray(curr_agent_seq[:, :-1], dtype=float), decimals=4)
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx # First frame 
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1 # Last frame
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    classEncoding = np.asarray(config.one_hot_encoding[curr_agent_seq[-1][0]], dtype=float)
                    curr_agent_seq = np.array(curr_agent_seq[:-1], dtype=float)
                    curr_agent_seq = curr_agent_seq/10  # scaling factor： 10
                    
                    if ((curr_agent_seq.shape[1] != 20) or (pad_end - pad_front != 20)): # if the seq_len != 20, ignore
                        continue

                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1] #

                    _idx = num_agents_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq
                    curr_seq_class[_idx] = classEncoding

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_agent_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_agents_considered += 1
                
                if num_agents_considered > min_agent:
                    non_linear_ped += _non_linear_ped
                    num_agents_in_seq.append(num_agents_considered) 
                    loss_mask_list.append(curr_loss_mask[:num_agents_considered])
                    seq_list.append(curr_seq[:num_agents_considered])  
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    seq_list_class.append(curr_seq_class[:num_agents_considered])

        self.num_seq = len(seq_list)
        if not (np.array_equal(seq_list, [])):
            seq_list = np.concatenate(seq_list, axis=0) 
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            seq_list_class = np.concatenate(seq_list_class, axis=0) 
            loss_mask_list = np.concatenate(loss_mask_list, axis=0) 
            non_linear_ped = np.asarray(non_linear_ped)
            # Convert numpy -> Torch Tensor
            self.obs_classes = torch.tensor(np.stack(seq_list_class)).type(torch.float)
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :8]).type(torch.float) 
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, 8:]).type(torch.float) 
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :8]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, 8:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float) 
            cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]

            # Convert to Graphs
            self.v_obs = []
            self.v_pred = []
            
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end)) 
            for ss in range(len(self.seq_start_end)): 
                pbar.update(1)

                start, end = self.seq_start_end[ss]
                v_1= seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], pos_enc=True)
                v_1 = [np.concatenate([v_1[i], self.obs_classes[start:end,:]],axis=1) for i in range(v_1.shape[0])]
                v_1 = np.array(v_1)
                v_1 = torch.from_numpy(v_1).type(torch.float)
                self.v_obs.append(v_1.clone())
                



                v_= seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :])
                v_ = [np.concatenate([v_[i], self.obs_classes[start:end,:]],axis=1) for i in range(v_.shape[0])]
                v_ = np.array(v_)
                v_ = torch.from_numpy(v_).type(torch.float)

                self.v_pred.append(v_.clone())
                
            pbar.close()
    
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index): # index is seq_index
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :], 
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], 
            self.non_linear_ped[start:end], self.loss_mask[start:end, :], 
            self.v_obs[index],self.v_pred[index]
        ]
        return out
