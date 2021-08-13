from typing import Union, Optional, Callable
import torch
import torch.nn.functional as F
from torch.nn import Parameter, ReLU, LeakyReLU
from torch_sparse import spspmm
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import GCNConv, GATConv, Sequential
from torch_geometric.nn.conv import FastRGCNConv, RGCNConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, softmax)
from torch_geometric.utils.repeat import repeat
import random

import sys
import os
sys.path.append(os.environ["GRAPH_SUM"])
sys.path.append("/u/helbling/GraphSummarization/src/external_code/allRank/")
import src.external_code.allRank.allrank.models.losses.neuralNDCG as neuralNDCG

def topk(score, indices, ratio, batch, min_score=None, tol=1e-7, num_output_sentences=None):
    if indices is None:
        indices = torch.arange(score.shape[0])
    assert score.shape == indices.shape

    sentence_ratio = ratio
    word_ratio = ratio
    num_nodes = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    #print("batch")
    #print(batch)
    #print("num nodes")
    #print(num_nodes)
    #print("score shape")
    #print(score.shape)
    #print("indices shape")
    #print(indices.shape)
    #print("cum num nodes")
    #print(cum_num_nodes)
    assert cum_num_nodes.shape == num_nodes.shape
    
    index  = torch.arange(batch.size(0), dtype=torch.long, device=score.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = score.new_full((batch_size * max_num_nodes, ),
                         torch.finfo(score.dtype).min)
    #print("index")
    #print(index)
    #print("dense x shape")
    #print(dense_x.shape)
    dense_x[index] = score
    dense_x = dense_x.view(batch_size, max_num_nodes)
    _, perm = dense_x.sort(dim=-1, descending=True)
    #print(_)
    #print(perm)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    if not num_output_sentences is None:
        k = torch.ones(batch_size) * num_output_sentences
        k = k.to(score.device)
        print(k.device)
        print(num_nodes.device)
        k = torch.min(k, num_nodes)
    elif isinstance(ratio, int):
        k = num_nodes.new_full((num_nodes.size(0), ), ratio)
        k = torch.min(k, num_nodes)
    else:
        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

    mask = [
        torch.arange(k[i], dtype=torch.long, device=score.device) +
        i * max_num_nodes for i in range(batch_size)
    ]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    # filter indices to be in range
    #in_range = torch.logical_and(perm >= 0, perm < indices.shape[0])
    #perm = perm[torch.nonzero(perm >= 0)]
    #perm = perm[torch.nonzero(perm < indices.shape[0])]
    indices_perm = indices[perm]

    return indices_perm

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i
    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def convert_y_to_onehot(y, num_sentences):
    """
        Converts the y vector to a one hot encoded vector
    """
    batch_size = num_sentences.shape[0]
    onehot_vectors = []
    for index in range(batch_size):
        current_num_sentences = num_sentences[index].int()
        current_y = y[index]
        current_y = current_y[torch.nonzero(current_y > 0)].squeeze().long()
        base_vector = torch.zeros(current_num_sentences).to(y.device)
        base_vector[current_y] = 1
        onehot_vectors.append(base_vector)

    return onehot_vectors

class TopKPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh, hidden=1024, gat=False):
        super(TopKPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.gat = gat
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden = hidden 
        if not gat:
            self.model = Sequential("x, edge_index, edge_weight", [
                (GCNConv(self.in_channels, self.hidden), "x, edge_index, edge_weight -> x"),
                LeakyReLU(inplace=True),
                (GCNConv(self.hidden, self.hidden // 2), "x, edge_index, edge_weight -> x"),
                LeakyReLU(inplace=True),
                (GCNConv(self.hidden // 2, 1), "x, edge_index, edge_weight -> x"),
                #LeakyReLU(inplace=True),
                #(GCNConv(self.hidden // 4, 1), "x, edge_index, edge_weight -> x"),
                LeakyReLU(inplace=True)
            ])
        else:
            self.model = Sequential("x, edge_index", [
                (GATConv(self.in_channels, self.hidden), "x, edge_index -> x"),
                LeakyReLU(inplace=True),
                (GATConv(self.hidden, self.hidden // 2), "x, edge_index -> x"),
                LeakyReLU(inplace=True),
                (GATConv(self.hidden // 2, 1), "x, edge_index -> x"),
                #LeakyReLU(inplace=True),
                #(GCNConv(self.hidden // 4, 1), "x, edge_index, edge_weight -> x"),
                LeakyReLU(inplace=True)
            ])

        

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None, num_sentences=None, num_output_sentences=3, is_last_layer=True):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        #attn = x if attn is None else attn
        #attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        #score = (attn * self.weight).sum(dim=-1)
        edge_weight = edge_attr.squeeze()
        if not self.gat:
            score = self.model(x, edge_index, edge_weight).squeeze()
        else:
            score = self.model(x, edge_index).squeeze()
        score = torch.nan_to_num(score)
        #score = softmax(score, batch)
        # get seperate permutation for sentences and the rest (word and document nodes)
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0).to(x.device)
        num_sentence_nodes = num_sentences.to(x.device) # the number of sentence nodes in each batch graph 
        cumulative_num_nodes  = torch.cat(
                [num_nodes.new_zeros(1),
                num_nodes.cumsum(dim=0)[:-1]]
            , dim=0).to(x.device)
        batch_size = num_nodes.size(0)
        # get the sentence perm
        sentence_indices = [torch.arange(num_sentence_nodes[i], device=x.device) + cumulative_num_nodes[i] for i in range(batch_size)]
        sentence_indices = torch.cat(sentence_indices).long()
        sentence_scores = score[sentence_indices]
        sentence_batch = [torch.ones(num_sentence_nodes[i]) * i for i in range(len(num_sentence_nodes))]
        sentence_batch = torch.cat(sentence_batch).long().to(x.device)
        sentence_perm = topk(sentence_scores, sentence_indices, self.ratio, sentence_batch, self.min_score, num_output_sentences=num_output_sentences)
        # get the other perm
        num_other_nodes = num_nodes - num_sentence_nodes
        if num_other_nodes.sum() > 0:
            other_indices = [torch.arange(num_other_nodes[i], device=x.device) + cumulative_num_nodes[i] + num_sentence_nodes[i] for i in range(batch_size)]
            other_indices = torch.cat(other_indices).long().to(x.device)
            other_scores = score[other_indices]
            other_batch = [torch.ones(num_other_nodes[i]) * i for i in range(len(num_other_nodes))]
            other_batch = torch.cat(other_batch).long().to(x.device)
            other_perm = topk(other_scores, other_indices, self.ratio, other_batch, self.min_score)
            # combine the other and sentence perms
            perm = torch.cat((sentence_perm, other_perm)) 
        else:
            perm = sentence_perm
        # process the batch
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm], sentence_scores, sentence_batch

class GraphUNetCoarsening(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        activation (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, activation=F.relu, gat=False, embedding_mapping=False):
        super(GraphUNetCoarsening, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.activation = activation
        self.embedding_mapping = embedding_mapping
        self.sum_res = sum_res
        self.loss = torch.nn.MSELoss()
        self.supervised_loss = torch.nn.BCELoss()
        self.triplet_loss = torch.nn.TripletMarginLoss()
        channels = hidden_channels
        embedding_channels = 1024
        # embedding function 
        if self.embedding_mapping:
            self.embedding_function = torch.nn.Linear(self.in_channels, embedding_channels) 
            self.inverse_embedding_function = torch.nn.Linear(embedding_channels, out_channels)
        else:
            self.embedding_function = torch.nn.Identity()
            self.inverse_embedding_function = torch.nn.Identity()
            embedding_channels = in_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        if gat:
            base_module = GATConv
        else:
            base_module = GCNConv
        self.down_convs.append(base_module(embedding_channels, channels))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i], gat=gat))
            self.down_convs.append(base_module(channels, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(base_module(in_channels, channels))
        self.up_convs.append(base_module(in_channels, embedding_channels))
        #self.output_conv = GCNConv(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

    """
        Computes the supervised ranking loss
    """
    def compute_supervised_ranking_loss(self, data, sentence_scores, sentence_batch):
        max_label_length = 10
        num_sentences = data.num_sentences
        y_all = data.y
        y_all = torch.reshape(y_all, (-1, max_label_length))
        batch_size = sentence_batch.max() + 1
        # go through each element
        loss = 0.0
        for index in range(batch_size):
            # compute output onehot vector
            current_indices = torch.nonzero(sentence_batch == index)
            scores = sentence_scores[current_indices].squeeze()
            output_onehot = torch.sigmoid(scores)
            # convert label to a onehot vector
            y = y_all[index]
            label_indices = y[torch.nonzero(y > 0)].long().squeeze()
            if len(label_indices.shape) == 0:
                label_indices = label_indices[None]
            current_num_sentences = num_sentences[index]
            label_onehot = torch.zeros(current_num_sentences).to(data.x.device).float()
            label_onehot[label_indices] = 1.0
            # compute the loss
            loss += self.supervised_loss(output_onehot, label_onehot)
 
        mean_loss = loss / batch_size
        return mean_loss
        
    """
        Computes the supervised loss
    """
    def compute_supervised_loss(self, data, sentence_scores, sentence_batch):
        max_label_length = 10
        num_sentences = data.num_sentences
        y_all = data.y
        y_all = torch.reshape(y_all, (-1, max_label_length))
        batch_size = sentence_batch.max() + 1
        # go through each element
        loss = 0.0
        for index in range(batch_size):
            # compute output onehot vector
            current_indices = torch.nonzero(sentence_batch == index)
            scores = sentence_scores[current_indices].squeeze()
            output_onehot = torch.sigmoid(scores)
            if len(output_onehot.shape) < 1:
                output_onehot = output_onehot.unsqueeze(0)
            # convert label to a onehot vector
            y = y_all[index]
            label_indices = y[torch.nonzero(y > 0)].long().squeeze()
            if len(label_indices.shape) == 0:
                label_indices = label_indices[None]
            current_num_sentences = num_sentences[index]
            label_onehot = torch.zeros(current_num_sentences).to(data.x.device).float()
            label_onehot[label_indices] = 1.0
            # compute the loss
            loss += self.supervised_loss(output_onehot, label_onehot)
 
        mean_loss = loss / batch_size
        return mean_loss

    """
        Computes a basic MSE between an input graph and a 
        reconstructed graph. 
    """
    def compute_unsupervised_loss(self, original_x, reconstructed_x, num_sentences=None, sentence_only=False, batch=None):
        if not num_sentences is None and sentence_only:
            x = original_x
            # computes the loss solely based on the sentence nodes
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0).to(x.device)
            num_sentence_nodes = num_sentences.to(x.device) # the number of sentence nodes in each batch graph 
            cumulative_num_nodes  = torch.cat(
                    [num_nodes.new_zeros(1),
                    num_nodes.cumsum(dim=0)[:-1]]
                , dim=0).to(x.device)
            batch_size = num_nodes.size(0)
            # get the sentence perm
            sentence_indices = [torch.arange(num_sentence_nodes[i], device=x.device) + cumulative_num_nodes[i] for i in range(batch_size)]
            sentence_indices = torch.cat(sentence_indices).long()
            # select the correct nodes
            original_sentence_x = original_x[sentence_indices]
            reconstructed_sentence_x = reconstructed_x[sentence_indices]
            loss = self.loss(original_sentence_x, reconstructed_sentence_x)
        else:
            loss = self.loss(original_x, reconstructed_x)
        return loss

    """
        Computes the basic triplet loss of a given set of output scores on
        a set of triplets constructed from the specific examples triplets
    """
    def compute_triplet_loss(self, sentence_scores, rankings, sentence_batch, batch, num_sentences, num_triplets=5, ndcg=True):
        if ndcg:
            return self.compute_neural_ndcg_loss(sentence_scores, rankings, sentence_batch, batch, num_sentences)
        # generate a list of triplets from the rankings
        triplets = []
        batch_size = sentence_batch.max() + 1
        total_triplet_loss = 0.0
        for index in range(batch_size):
            current_scores = sentence_scores[torch.nonzero(sentence_batch == index)]
            current_rankings = rankings[torch.nonzero(sentence_batch == index)]
            #current_rankings = current_rankings[0: current_rankings.shape[0]]
            assert current_scores.shape == current_rankings.shape
            num_scores = current_scores.shape[0]
            if num_scores <= 3:
                continue
            current_triplet_loss = 0.0
            for triplet_num in range(num_triplets):
                random_indices = random.sample(range(0, num_scores), 3)
                # sort the numbers as anchor, positive, negative
                anchor_rank = current_rankings[random_indices[0]]
                first_rank = current_rankings[random_indices[1]]
                second_rank = current_rankings[random_indices[2]]
                # is first closer than second
                first_closer_than_second = abs(anchor_rank - first_rank) < abs(anchor_rank - second_rank)
                if not first_closer_than_second:
                    # swap indices
                    temp = random_indices[1]
                    random_indices[1] = random_indices[2]
                    random_indices[2] = temp
                # evaluate the triplet
                anchor_score = current_scores[random_indices[0]].unsqueeze(0)
                left_score = current_scores[random_indices[1]].unsqueeze(0)
                right_score = current_scores[random_indices[2]].unsqueeze(0)
                # calculate the triplet margin loss of the scores
                triplet_loss = self.triplet_loss(anchor_score, left_score, right_score)
                current_triplet_loss +=  triplet_loss

            total_triplet_loss += current_triplet_loss / num_triplets

        return total_triplet_loss

    def compute_neural_ndcg_loss(self, sentence_scores, rankings, sentence_batch, batch, num_sentences):
        batch_size = sentence_batch.max() + 1
        slate_size = 200 # max num sentences
        # reshape the inputs
        stacked_scores = []
        stacked_rankings = []
        for index in range(batch_size):
            # sentence_scores should be [batch_size, slate_size]
            # rankings should be [batch_size, slate_size] 
            current_scores = sentence_scores[torch.nonzero(sentence_batch == index)]
            current_rankings = rankings[torch.nonzero(sentence_batch == index)]
            # pad the scores and rankings with -1 to have the desired size
            if current_scores.shape[0] == 1:
                continue
            current_scores = current_scores.squeeze()
            current_scores_shape = current_scores.shape[0]
            current_scores = F.pad(input=current_scores, pad=(0, slate_size - current_scores_shape), mode='constant', value=-1)
            stacked_scores.append(current_scores)
            current_rankings = current_rankings.squeeze()
            current_rankings_shape = current_rankings.shape[0]
            current_rankings = F.pad(input=current_rankings, pad=(0, slate_size - current_rankings_shape), mode='constant', value=-1)
            stacked_rankings.append(current_rankings.squeeze())
        stacked_scores = torch.stack(stacked_scores)
        stacked_rankings = torch.stack(stacked_rankings)
        assert stacked_scores.shape == stacked_rankings.shape
        # calculate loss
        loss = -1 * neuralNDCG(stacked_scores, stacked_rankings, padded_value_indicator=-1)
        return loss 

    """
        def forward(self, input_graph, num_output_sentences=3):
            # unpack input
            data_list = input_graph.to_data_list()
            loss = 0.0
            coarse_indices = []
            for data in data_list:
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                edge_weight = edge_attr.squeeze().float()
                num_sentences = data.num_sentences
                y = data.y
                # run the layers
                x = self.layer_one(x, edge_index, edge_weight)
                output_attention = self.layer_two(x, edge_index, edge_weight).squeeze()
                assert output_attention.shape[0] == x.shape[0]
                # convert the attention to a onehot vector
                sentence_attention = output_attention[0:num_sentences]
                # get topk
                num_output_nodes = torch.nonzero(y >= 0).shape[0]
                if num_output_nodes == 0:
                    num_output_nodes = 1
                topk_values, topk_indices = torch.topk(sentence_attention, num_output_nodes)
                cutoff = topk_values[-1]
                # get all coarse indices
                all_topk_values, all_topk_indices = torch.topk(output_attention, x.shape[0])
                coarse_inds = all_topk_indices[torch.nonzero(all_topk_values >= cutoff).squeeze()]
                coarse_indices.append(coarse_inds)
                # compute output 
                output_onehot = torch.sigmoid(sentence_attention)
                # convert label to a onehot vector
                label_indices = y[torch.nonzero(y > 0)].long().squeeze()
                if len(label_indices.shape) == 0:
                    label_indices = label_indices[None]
                label_onehot = torch.zeros(num_sentences).to(device).float()
                label_onehot[label_indices] = 1.0
                # compute the loss
                loss += self.supervised_loss(output_onehot, label_onehot)
    """         
 
    """
        Performs forward pass 
    """
    def forward(self, data, num_output_sentences=3, batch=None, mode="supervised"):
        if batch is None:
            batch = data.batch
        original_batch = batch.clone()
        # unpack the data
        x, y, edge_index, edge_attr, num_sentences = data.x, data.y, data.edge_index, data.edge_attr, data.num_sentences
        rankings = data.rankings
        # convert edge_attr to edge_weight
        edge_weight = edge_attr.squeeze().float()
        edge_index = edge_index.long()
        x = x.float()
        # save a copy of the input x
        original_x = x.clone()
        # do input embedding function
        x = self.embedding_function(x)
        # perform first layer down convolution 
        x = self.down_convs[0](x, edge_index)
        x = self.activation(x)
        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []
        last_batch = None
        # perform downward pooling and convolutions
    
        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            is_last_layer = not i < self.depth
            x, edge_index, edge_weight, batch, perm, perm_scores, sentence_scores, sentence_batch = self.pools[i - 1](
                                                                     x,
                                                                     edge_index,
                                                                     edge_attr=edge_weight, 
                                                                     num_sentences=num_sentences, 
                                                                     num_output_sentences=num_output_sentences, 
                                                                     batch=batch,
                                                                     is_last_layer=is_last_layer)

            x = self.down_convs[i](x, edge_index)
            x = self.activation(x)
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
            last_batch = batch
        coarsened_indices = perms[-1]
        # perform upward pooling and convolutions
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index)
            x = self.activation(x) if i < self.depth - 1 else x
        # output embedding
        x = self.inverse_embedding_function(x)
        # compute the loss
        reconstructed_x = x
        supervised_loss = self.compute_supervised_loss(data, sentence_scores, sentence_batch)
        unsupervised_loss = self.compute_unsupervised_loss(original_x, reconstructed_x, num_sentences=num_sentences, batch=original_batch)
        triplet_loss = self.compute_triplet_loss(sentence_scores, rankings, sentence_batch, batch, num_sentences)
        if mode == "supervised":
            loss = supervised_loss
        elif mode == "triplet":
            loss = triplet_loss
        elif mode == "unsupervised":
            loss = unsupervised_loss
        # make a loss dictionary
        loss_dict = {
            "supervised_loss": supervised_loss,
            "unsupervised_loss": unsupervised_loss,
            "triplet_loss": triplet_loss, 
        }

        return loss_dict, loss, coarsened_indices
