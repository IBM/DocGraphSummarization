##Method2: Directly compare original features and coarsed features (with the same dimension,
# GNN(fe, hidden) - Coarsen - GNN(hidden, fe) ).
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, JumpingKnowledge
from sinkhorn import sinkhorn_loss_default
from pytorch_memlab import profile, set_target_gpu, profile_every
import numpy as np

class GNNBlock(torch.nn.Module): #2 layer GCN block
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNBlock, self).__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, out_channels)

        self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                   out_channels)
        #self.lin1 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        return self.lin(torch.cat([x1, x2], dim=-1))
        # return self.lin1(x1, dim=-1)

class CoarsenBlock(torch.nn.Module):
    def __init__(self, in_channels, assign_ratio):
        super(CoarsenBlock, self).__init__()
        self.gcn_att = DenseGCNConv(in_channels, 1, bias=True)
        # self.att = torch.nn.Linear(in_channels,
        #                            hidden)
        self.assign_ratio = assign_ratio

    def normalize_batch_adj(self, adj):  # adj shape: batch_size * num_node * num_node, D^{-1/2} (A+I) D^{-1/2}
        dim = adj.size()[1]
        A = adj + torch.eye(dim, device=adj.device)
        deg_inv_sqrt = A.sum(dim=-1).clamp(min=1).pow(-0.5)
        newA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        newA = (adj.sum(-1)>0).float().unsqueeze(-1).to(adj.device) * newA
        return newA

    def reset_parameters(self):
        self.gcn_att.reset_parameters()

    def get_topk_range_for_sentences(self, alpha_vec, output_node_count, num_sentences):
        assert num_sentences >= output_node_count
        alpha_vec = alpha_vec.detach().cpu().numpy()
        full_argsort = np.argsort(alpha_vec)[::-1]
        topk_ind = [] 
        num_sentences = 0
        for index in full_argsort:
            if num_sentences == output_node_count:
                break
            is_sentence = index < num_sentences 
            if is_sentence:
                num_sentences += 1
            topk_ind.append(index)
        
        temptopk = torch.Tensor(alpha_vec[topk_ind])
        topk_ind = torch.Tensor(topk_ind.copy()).int()
        # return temptopk and topkind 
        return temptopk, topk_ind
        """
        assert num_sentences > output_node_count
        alpha_vec = alpha_vec.detach().cpu().numpy()
        # get threshold for sentence cutoff
        sentence_alpha_vec = alpha_vec[0: num_sentences]
        sentence_alpha_vec_sorted_args = np.argsort(sentence_alpha_vec)[::-1]
        print("output node count - 1")
        print(output_node_count - 1)
        index_of_nth_sentence = sentence_alpha_vec_sorted_args[output_node_count - 1]
        topk_ind = []
        full_argsort = np.argsort(alpha_vec)[::-1]
        for index in full_argsort:
            topk_ind.append(index)
            if index == index_of_nth_sentence:
                break
        # get nodes above threshold
        # finde index of nth sentence 
        #index_of_index_of_nth_sentence = np.where(index_of_nth_sentence == full_argsort)[0][0]
        #print("index of index")
        #print(index_of_index_of_nth_sentence)
        # get all indices up until that point
        #topk_ind = full_argsort[0: index_of_index_of_nth_sentence + 1]
        temptopk = torch.Tensor(alpha_vec[topk_ind])
        topk_ind = torch.Tensor(topk_ind.copy()).int()
        # return temptopk and topkind 
        return temptopk, topk_ind
        """

    def forward(self, x, adj, batch_num_nodes, num_sentences=None, output_node_counts=None):
        alpha_vec = F.sigmoid(torch.pow(self.gcn_att(x,adj),2)).squeeze() # b*n*1 --> b*n
        if len(alpha_vec.shape) < 2:
            alpha_vec = alpha_vec.unsqueeze(0)
        norm_adj = self.normalize_batch_adj(adj)
        batch_size = x.size()[0]
        cut_batch_num_nodes = batch_num_nodes
        if len(alpha_vec.shape) < 2:
            cut_value = torch.zeros_like(alpha_vec) 
        else:
            cut_value = torch.zeros_like(alpha_vec[:, 0])
        batch_topk_ind = []
        for j in range(batch_size):
            if cut_batch_num_nodes[j] > 1:
                cut_batch_num_nodes[j] = torch.ceil(cut_batch_num_nodes[j].float() * self.assign_ratio)+1
                # cut_value[j], _ = (-alpha_vec[j]).kthvalue(cut_batch_num_nodes[j], dim=-1)
                if output_node_counts is None:
                    temptopk, topk_ind = alpha_vec[j].topk(cut_batch_num_nodes[j].int().item(), dim=-1)
                else:
                    current_alpha_vec = alpha_vec[j]
                    temptopk, topk_ind = self.get_topk_range_for_sentences(current_alpha_vec, output_node_counts[j].int().item(), num_sentences)
                cut_value[j] = temptopk[-1]
                batch_topk_ind.append(topk_ind)
            else:
                cut_value[j] = 0
        # cut_alpha_vec = torch.mul( ((alpha_vec - torch.unsqueeze(cut_value, -1))>=0).float(), alpha_vec)  # b * n
        cut_alpha_vec = F.relu(alpha_vec+0.0000001 - torch.unsqueeze(cut_value, -1))
        S = torch.mul(norm_adj, cut_alpha_vec.unsqueeze(1))  # repeat rows of cut_alpha_vec, #b * n * n
        # temp_rowsum = torch.sum(S, -1).unsqueeze(-1).pow(-1)
        # # temp_rowsum[temp_rowsum > 0] = 1.0 / temp_rowsum[temp_rowsum > 0]
        # S = torch.mul(S, temp_rowsum)  # row-wise normalization
        S = F.normalize(S, p=1, dim=-1)
        embedding_tensor = torch.matmul(torch.transpose(S, 1, 2),
                                        x)  # equals to torch.einsum('bij,bjk->bik',...)
        new_adj = torch.matmul(torch.matmul(torch.transpose(S, 1, 2), adj), S)  # batched matrix multiply

        return embedding_tensor, new_adj, S, batch_topk_ind

class Coarsening(torch.nn.Module):
    def __init__(self, dataset, hidden, ratio=0.5, epsilon=1.0, opt_epochs=10): # we only use 1 layer for coarsening
        super(Coarsening, self).__init__()
        self.ratio = ratio
        self.epsilon = epsilon
        self.opt_epochs = opt_epochs
        # self.embed_block1 = GNNBlock(dataset.num_features, hidden, hidden)
        self.embed_block1 = DenseGCNConv(dataset.num_features, hidden)
        self.coarse_block1 = CoarsenBlock(hidden, ratio)
        self.embed_block2 = DenseGCNConv(hidden, dataset.num_features)
        self.jump = JumpingKnowledge(mode='cat')

        # self.lin1 = Linear(hidden + dataset.num_features, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.coarse_block1.reset_parameters()
        self.jump.reset_parameters()
        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()

    def compute_loss(self, x, x2, epsilon, opt_epochs, p):
        opt_loss = 0.0
        for i in range(len(x)):
            x3 = self.get_nonzero_rows(x[i])
            x4 = self.get_nonzero_rows(x2[i])
            # if x3.size()[0]==0 or x4.size()[0]==0:
            #     continue
            # opt_loss += sinkhorn_loss_default(x3, x4, epsilon, niter=opt_epochs).float()
            opt_loss += sinkhorn_loss_default(x3, x2[i], epsilon, niter=opt_epochs, p=p)

        return opt_loss

    #@profile_every(1)
    def forward(self, data, p=2, output_node_counts=None, num_sentences=None):
        x, adj, mask = data.x, data.adj, data.mask
        batch_num_nodes = data.mask.sum(-1)
        x1 = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        # xs = [x1.mean(dim=1)]
        coarse_x, new_adj, S, batch_topk_ind = self.coarse_block1(x1, adj, batch_num_nodes, num_sentences=num_sentences, output_node_counts=output_node_counts)
        xs = [coarse_x.mean(dim=1)]
        x2 = F.tanh(self.embed_block2(coarse_x, new_adj, mask, add_loop=True))
        xs.append(x2.mean(dim=1))

        opt_loss = self.compute_loss(x, x2, self.epsilon, self.opt_epochs, p)

        return xs, new_adj, S, opt_loss, batch_topk_ind

    def get_nonzero_rows(self, M):# M is a matrix
        # row_ind = M.sum(-1).nonzero().squeeze() #nonzero has bugs in Pytorch 1.2.0.........
        #So we use other methods to take place of it
        MM, MM_ind = M.sum(-1).sort()
        N = (M.sum(-1)>0).sum()
        return M[MM_ind[:N]]

    def __repr__(self):
        return self.__class__.__name__

class MultiLayerCoarsening(torch.nn.Module):
    def __init__(self, dataset, hidden, num_layers=2, ratio=0.5):
        super(MultiLayerCoarsening, self).__init__()

        self.embed_block1 = DenseGCNConv(dataset.num_features, hidden)
        self.coarse_block1 = CoarsenBlock(hidden, ratio)
        self.embed_block2 = DenseGCNConv(hidden, dataset.num_features)
        # self.embed_block2 = GNNBlock(hidden, hidden, dataset.num_features)

        self.num_layers = num_layers

        self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear( hidden + dataset.num_features, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.coarse_block1.reset_parameters()
        self.embed_block2.reset_parameters()

        self.jump.reset_parameters()
        # self.lin1.reset_parameters()

    """
        Compute loss for each example
    """
    def compute_loss(self, x, x2, epsilon, out_epochs):
        opt_loss = 0.0
        for i in range(len(x)):
            x3 = self.get_nonzero_rows(x[i])
            x4 = self.get_nonzero_rows(x2[i])
            if x3.size()[0]==0:
                continue
            if x4.size()[0]==0:
                # opt_loss += sinkhorn_loss_default(x3, x2[i], epsilon, niter=opt_epochs).float()
                continue
            opt_loss += sinkhorn_loss_default(x3, x4, epsilon, niter=opt_epochs).float()
        
        return opt_loss
    
    def get_original_graph_coarse_indices(self, batch_topks):
        """
            Returns a tensor with each index corresponding to the
            a node in the coarse graph. The value at the index is the
            index of the original node corresponding to the coarse node. 
            
            parameters
                - batch_topks: This is a list with length "num layers" with each element "batch_size" where
                 each element contains a list of indices corresponding to the nodes 
                in the coarse graph. The value at each index is the node in the original 
                graph the coarse node maps to. 
                
                Example: topk_ind[i] = j means node i in the coarse graph maps to node j in 
                the original graph.  
        """
        # shape is (num_layers, batch_size, num_coarse_nodes)  
        # transpose num_layers and batch_size 
        batch_topks = [list(i) for i in zip(*batch_topks)] 
        batch_coarse_indices = []
        # shape is (batch_size, num_layers, num_coarse_nodes)
        for batch_index, layerwise_topks in enumerate(batch_topks):
            coarse_indices = layerwise_topks[-1]
            for layer_num in reversed(range(len(layerwise_topks) - 1)):
                topks = layerwise_topks[layer_num]
                coarse_indices = [topks[coarse_index] for coarse_index in coarse_indices]
            # should be the same size as the original coarse_indices with indices
            # domain of the original graph
            batch_coarse_indices.append(coarse_indices)
    
        return batch_coarse_indices

    def forward(self, data, epsilon=0.01, opt_epochs=100):
        x, adj, mask = data.x, data.adj, data.mask
        batch_num_nodes = data.mask.sum(-1)
        batch_topks = []
        new_adjs = [adj]
        Ss = []
        x1 = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x1.mean(dim=1)]
        new_adj = adj
        coarse_x = x1
        # coarse_x, new_adj, S = self.coarse_block1(x1, adj, batch_num_nodes)
        # new_adjs.append(new_adj)
        # Ss.append(S)

        for i in range(self.num_layers):
            coarse_x, new_adj, S, batch_topk_ind = self.coarse_block1(coarse_x, new_adj, batch_num_nodes)
            new_adjs.append(new_adj)
            Ss.append(S)
            batch_topks.append(batch_topk_ind)
        x2 = self.embed_block2(coarse_x, new_adj, mask, add_loop=True)#should not add ReLu, otherwise x2 could be all zero.
        xs.append(x2.mean(dim=1))
        # compute loss 
        opt_loss = self.compute_loss(x, x2, epsilon, out_epochs)
        coarse_indices = self.get_original_graph_coarse_indices(batch_topks)
        return xs, new_adjs, Ss, opt_loss, coarse_indices

    def get_nonzero_rows(self, M):# M is a matrix
        # row_ind = M.sum(-1).nonzero().squeeze() #nonzero has bugs in Pytorch 1.2.0.........
        # So we use other methods to take place of it
        MM, MM_ind = torch.abs(M.sum(-1)).sort()
        N = (torch.abs(M.sum(-1)) > 0).sum()
        return M[MM_ind[:N]]

    def __repr__(self):
        return self.__class__.__name__
