import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from layers import *
from layers.gcn_family import *
from layers.GATv2 import GATv2Conv
from utils.tools import positional_encoding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
torch.manual_seed(0)
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PedestrianPredictionFormer(nn.Module):

    def __init__(self, in_size, obs_len, pred_len, embed_size, patch_len, stride, single_placeholder):
        super(PedestrianPredictionFormer, self).__init__()

        self.embedding = nn.Linear(in_size*(obs_len + pred_len), embed_size)
        self.cls_head = nn.Linear(embed_size, 1)
        self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
        self.reg_head = nn.Linear(embed_size, in_size*pred_len)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.single_step_placeholder = single_placeholder
        self.embed_size = embed_size
        
        self.feature_size = 32
        self.embedding_norm = nn.Linear(1, int(self.feature_size))
        self.embedding_angle = nn.Linear(1, int(self.feature_size))
        
        self.embedding_traj = nn.Linear(2, int(self.feature_size))
        self.embedding_vel = nn.Linear(2, int(self.feature_size)*2)
        
        
        self.context_window = 8
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((self.context_window - self.patch_len)/self.stride + 1)
        self.traj_embedding = nn.Linear(2, embed_size)
        
        # NOTE: Initialization HodgeLaguerreConv
        self.K = 1
        self.EdgeEmbed = nn.Linear(1, embed_size)
        self.HodgeGCN = HodgeLaguerreConv(in_channels=embed_size, out_channels=embed_size, K=self.K, bias=True)
        self.EdgeCausal = nn.Linear(embed_size, self.K*2)
        
        self.NodeGCN = GCNConv(in_channels=embed_size, out_channels=embed_size, K=self.K)
        self.NodeGAT = GATv2Conv(in_channels=embed_size, out_channels=embed_size, add_self_loops=False)
        
        self.traj_transform = nn.Linear(embed_size, 2)
        self.ln = nn.LayerNorm(embed_size)
        
        # NOTE: Initialization Transformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=0.1,
                                      output_attention=True), embed_size, 4),
                    embed_size,
                    256,
                    dropout=0.01,
                    activation='gelu'
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(embed_size)
        )
        
        pe='zeros'
        learn_pe=True
        self.W_pos = positional_encoding(pe, learn_pe, (self.patch_num + self.pred_len), embed_size)
        self.dropout = nn.Dropout(p=0.05, inplace=False)

        self.output = nn.Linear(embed_size, 5)
        self.placeholder_initialized = False
       

    def spatial_interaction(self, ped, edge_index, edge_attr, edge_edge_index):
            
        z = ped.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = rearrange(z, 'n l d t -> l n t d')     
        z1 = rearrange(z, 'l n t d -> l (n t) d') 

        graph_features = []
        for i in range(z1.shape[0]):

            edge_feature = edge_attr[i]
            if torch.isnan(edge_feature).any() or torch.isinf(edge_feature).any():
                edge_feature[torch.isnan(edge_feature) | torch.isinf(edge_feature)] = 0

            edge_embedding = self.EdgeEmbed(edge_attr[i].float().unsqueeze(1).to(device))
            num_edges = edge_edge_index[i].shape[1]  
            initial_edge_weight = torch.ones(num_edges)
            node_embedding, (adj, edge_weight) = self.NodeGAT(z1[i], edge_index[i].to(device), edge_attr=None, return_attention_weights=True)
            edge_updated = self.HodgeGCN(edge_embedding, edge_edge_index[i].to(device), initial_edge_weight)
            edge_norm = self.EdgeCausal(edge_updated)
            node_embedding = self.NodeGCN(node_embedding, edge_index[i].to(device), edge_norm)
            node_updated = torch.add(node_embedding.squeeze(), z1[i])
            graph_features.append(node_updated)
        # ------------------------------------------------------------------------------------ #
        graph_features = torch.stack(graph_features, dim=0)
        graph_recovery = rearrange(graph_features, 'l (n t) d -> l n t d', t=self.patch_len) 
        return graph_recovery
    
    def forward(self, ped_obs, vel, nodes, edge_index, edge_attr, edge_edge_index):
        assert len(ped_obs.shape) == 3              
        assert len(nodes.shape) == 2           

        num_people = ped_obs.shape[0]
        self.single_step_placeholder = nn.Parameter(torch.randn(1, num_people, self.embed_size))
        self.register_parameter('step_placeholder', self.single_step_placeholder)
      
        vel_pre = torch.zeros_like(vel)
        vel_pre[:,1:] = vel[:,:-1]  
        vel_pre[:,0] = vel[:,0]
        EPS = 1e-6
        vel_cosangle = torch.sum(vel_pre*vel,dim=-1)/((torch.norm(vel_pre,dim=-1)+EPS)*(torch.norm(vel,dim=-1)+EPS)) 
        vel_angle = torch.acos(torch.clamp(vel_cosangle,-1,1))                                                       
        
        # =========================================
        norm = self.embedding_norm(nodes.unsqueeze(-1)) 
        angle = self.embedding_angle(vel_angle.unsqueeze(-1))                                       
        normAngle = torch.cat([norm,angle],dim=-1)                                          
        vel_embed = self.embedding_vel(vel)
        node_features = torch.cat([normAngle,vel_embed],dim=-1)                          
        graph_recovery = self.spatial_interaction(node_features, edge_index, edge_attr, edge_edge_index)        
        graph_trajectory = reduce(graph_recovery, 't n l d ->t n d', 'mean')                   
        graph_trajectory = self.dropout(graph_trajectory.permute(1,0,2))      

        # Adding placeholder
        placeholder = torch.repeat_interleave(self.single_step_placeholder.permute(1,0,2), self.pred_len, dim=1).to(device)
        if len(placeholder.shape) != 3:
            trajectories = torch.cat([graph_trajectory, placeholder.unsqueeze(0)], dim=1) 
        trajectories = torch.cat([graph_trajectory, placeholder], dim=1)                       
        
        trajectories = self.dropout(trajectories + self.W_pos)                                 
        
        prediction, attns = self.encoder(trajectories, attn_mask=None)

        prediction = self.output(prediction)                                                   

        return prediction.contiguous()
        
        
        



















