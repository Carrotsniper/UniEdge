import pickle
import torch
from torch.utils.data.dataloader import DataLoader
import random
import argparse
from utils.utils_preprocess import *
from utils.util import *
from einops import rearrange
from utils.metrics import * 
from utils.config import *
from utils.tools import *   
import torch.nn.functional as F
from test_utils import *
from tqdm import tqdm   
from model import PedestrianPredictionFormer
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='eth')
args = parser.parse_args()

@torch.no_grad()
def test(model, loader_test,  rel_trajectory, abs_trajectory, patch_len, stride, pred_len, KSTEPS=20):

    model.eval()
    ade_all, fde_all = [], []

    for batch in tqdm(loader_test, desc="Processing batches"): 
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        full_rel = torch.cat((obs_traj_rel.squeeze(0).permute(0,2,1), pred_traj_gt_rel.squeeze(0).permute(0,2,1)), dim=1)
        full_abs = torch.cat((obs_traj.squeeze(0).permute(0,2,1), pred_traj_gt.squeeze(0).permute(0,2,1)), dim=1)
        
        angles = [0]
        end_error, rst = configure(
            full_rel.cpu().numpy(),
            full_abs.cpu().numpy(),
            rel_trajectory.cpu().numpy(),
            abs_trajectory.cpu().numpy(),
            angles,
            option=2,)
        rst = torch.stack(rst)  
        dest = rst.unsqueeze(1).reshape(obs_traj.shape[1], 1, 2).repeat(1, 8, 1)
        V_obs_new = V_obs.squeeze(0).permute(1,0,2)[:,:,1:] - dest
        num = V_obs[:,:, :, 0].unsqueeze(2).permute(0,1,3,2)
        V_obs = torch.cat([num, V_obs_new.unsqueeze(0).permute(0,2,1,3)], dim=3)

        absolute_trajectory = rearrange(obs_traj.squeeze(0), 'n d t -> n t d')
        
        edge_index, edge_attr, edge_edge_index  = pdd_matrix_edges(absolute_trajectory, patch_len, stride, 2)

        relative_trajectory = rearrange(V_obs.squeeze(0), "t n d -> n t d")
        nodes = torch.sqrt(torch.sum(relative_trajectory[:,:,1:]**2, dim=-1)).detach()

        prediction = model(absolute_trajectory.float().to(device), relative_trajectory[:, :, 1:].float().to(device), nodes.float().to(device), edge_index, edge_attr, edge_edge_index)
        prediction = prediction.permute(1,0,2)[-pred_len:, :, :]
       
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)              # torch.Size([8, 1, 2])
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)      # torch.Size([12, 1, 2])
        mu, cov = generate_statistics_matrices(prediction)
        ade_stack, fde_stack = [], []
        for trial in range(5):
            sample_level = 'scene'
            if sample_level == 'scene':
                r_sample = random.randn(1, KSTEPS, 2)
            else:
                raise NotImplementedError

            r_sample = torch.Tensor(r_sample).to(dtype=mu.dtype, device=mu.device)
            r_sample = r_sample.permute(1, 0, 2).unsqueeze(dim=1).expand((KSTEPS,) + mu.shape)
            V_pred_sample = mu + (torch.linalg.cholesky(cov) @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)

            V_absl = V_pred_sample.cumsum(dim=1) + V_obs_traj[[-1], :, :]
            ADEs, FDEs = compute_batch_metric(V_absl, V_pred_traj_gt)

            ade_stack.append(ADEs.detach().cpu().numpy())
            fde_stack.append(FDEs.detach().cpu().numpy())
           
        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))

    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    mean_ade, mean_fde = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean()
    return mean_ade, mean_fde


def main():
    paths = './checkpoint/eth/eth_ckpt'
    print(paths)
    print("*" * 50)
    print("Evaluating model:", paths)

    model_path =  paths + '/val_best.pth'
    args_path = paths + '/args.pkl'
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    ADE_ls, FDE_ls = [], []
    KSTEPS = 20
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    # Data prep
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './datasets/'+ args.dataset_name + '/'
    with open('./dataset_preprocessed/eth_processed.pkl', 'rb') as f:
        dset_train = pickle.load(f)

    loader_train = DataLoader(
        dset_train,
        batch_size=1, 
        shuffle=True,
        num_workers=0)

    rel_trajectory, abs_trajectory = train_data_processor(loader_train)

    dset_test = TrajectoryDatasetTest(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    model = PedestrianPredictionFormer(in_size=3, obs_len=args.obs_len, pred_len=args.pred_len, embed_size=128, patch_len=args.patch_len, stride=args.stride, single_placeholder=True).to(device)

    model.load_state_dict(torch.load(model_path), strict=False)
    print("Checkpoint Loaded: ", model_path)
    print("Testing ....")
    ADE, FDE = test(model, loader_test, rel_trajectory, abs_trajectory, args.patch_len, args.stride, args.pred_len)
    ADE_ls.append(ADE)
    FDE_ls.append(FDE)
    
    print("Scene: {} ADE: {:.8f} FDE: {:.8f}".format('eth', ADE, FDE))
    


if __name__ == '__main__':
    main()
