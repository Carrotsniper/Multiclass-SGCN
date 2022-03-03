import torch
import torch.nn as nn
from torch.nn import functional as F


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):

        shortcut = self.shortcut(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2 + shortcut


class InteractionMask(nn.Module):

    def __init__(self, number_asymmetric_conv_layer=7, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction):

        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        
        temporal_mean = torch.mean(temporal_interaction_mask, dim=1)
        temporal_zero = torch.zeros_like(temporal_interaction_mask[0],device='cuda')
        spatial_mean = torch.mean(spatial_interaction_mask, dim=1)
        spatial_zero = torch.zeros_like(spatial_interaction_mask[0],device='cuda')


        tem = torch.clone(temporal_interaction_mask)
        spa = torch.clone(spatial_interaction_mask)

        for index, value in enumerate(spatial_mean):
            spa[index, :] = torch.where(spatial_interaction_mask[index,:] > value, spatial_interaction_mask[index,:], spatial_zero)
        
        for index, value in enumerate(temporal_mean):
            tem[index, :] = torch.where(temporal_interaction_mask[index,:] > value, temporal_interaction_mask[index,:], temporal_zero)


        spatial_interaction_mask1 = torch.clone(spa)
        temporal_interaction_mask1 = torch.clone(tem)
        return spatial_interaction_mask1, temporal_interaction_mask1


class ZeroSoftmax(nn.Module):

     def __init__(self):
         super(ZeroSoftmax, self).__init__()

     def forward(self, x, dim=0, eps=1e-5):
         x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
         x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
         x = x_exp / (x_exp_sum + eps)
         return x

class SelfAttention(nn.Module):
  
    def __init__(self,in_dims=8, ind1=2, ind2=6, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()

        self.embedding1 = nn.Linear(ind1, d_model)
        self.embedding2 = nn.Linear(ind2, d_model)
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads

    def split_heads(self, x):

        # x [batch_size seq_len d_model]

        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()

        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]

    def forward(self, x, mask=False, multi_head=False):

        # batch_size seq_len 2

        assert len(x.shape) == 3

        if x.shape[2] == 8:
            embeddings1 = self.embedding1(x[:,:,:2])  # batch_size seq_len d_model
            embeddings2 = self.embedding2(x[:,:,2:])
        elif x.shape[2] ==9:
            embeddings1 = self.embedding1(x[:,:,:3])  # batch_size seq_len d_model
            embeddings2 = self.embedding2(x[:,:,3:])

        embeddings = embeddings1 + embeddings2

       
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:

            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)

        return attention, embeddings


class SpatialTemporalFusion(nn.Module):

    def __init__(self, obs_len=8):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential()
    def forward(self, x):

        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


class SparseWeightedAdjacency(nn.Module):

    def __init__(self, SPA_in_dims=8, TEM_in_dims=9, spa_in=2, tem_in=3, labels_enc=6, embedding_dims=64, obs_len=8, dropout=0,
                 number_asymmetric_conv_layer=7):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        self.spatial_attention = SelfAttention(SPA_in_dims,spa_in, labels_enc, embedding_dims)
        self.temporal_attention = SelfAttention(TEM_in_dims,tem_in, labels_enc, embedding_dims)

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        self.dropout = dropout
        # self.zero_softmax = nn.Softmax()
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):

        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)

        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)

        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True)

        # attention fusion
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)

        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,\
               spatial_embeddings, temporal_embeddings


class GraphConvolution(nn.Module):

    def __init__(self, in_dims=8, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):

        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):

        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]

        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)

        # [num_p num_heads seq_len d]
        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn[1](gcn_spatial_features, normalized_temporal_adjacency_matrix)

        gcn_temporal_features = self.temporal_spatial_sparse_gcn[0](tem_graph,
                                                                   normalized_temporal_adjacency_matrix)
        gcn_temporal_features = gcn_temporal_features.permute(2, 1, 0, 3)
        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn[1](gcn_temporal_features,
                                                                            normalized_spatial_adjacency_matrix)

        return gcn_spatial_temporal_features, gcn_temporal_spatial_features.permute(2, 1, 0, 3)


class TrajectoryModel_attn(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel_attn, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learningc
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            SPA_in_dims=8, TEM_in_dims=9,spa_in=2, tem_in=3, labels_enc=6,number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
            in_dims=8, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()
        ))

        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()
        ))

        self.output = nn.Linear(embedding_dims // num_heads, out_dims)

    def forward(self, graph, identity):

        # graph 1 obs_len N 3

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)

        gcn_temporal_spatial_features, gcn_spatial_temporal_features = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )

        gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features

        gcn_representation = gcn_representation.permute(0, 2, 1, 3)

        features = self.tcns[0](gcn_representation)

        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        prediction = torch.mean(self.output(features), dim=-2)

        return prediction.permute(1, 0, 2).contiguous()
