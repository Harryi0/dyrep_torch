# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

import os.path as osp
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from datetime import datetime, timedelta
from collections import defaultdict
from torch.nn import Linear, Parameter
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage,
                                           LastAggregator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0].to(device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
min_src_idx, max_src_idx = int(data.src.min()), int(data.src.max())

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)
train_data_small = data[:20]

def normalize_td(data):
    dic = defaultdict(list)
    src, dst, t = data.src.numpy(), data.dst.numpy(), data.t.numpy()
    for i in range(len(src)):
        dic[src[i]].append(t[i])
        dic[dst[i]].append(t[i])
    all_diff = []
    all_td = []
    for k, v in dic.items():
        ts = np.array(v)
        td = np.diff(ts)
        all_diff.append(td)
        if len(v) >= 2:
            timestamp = np.array(list(map(lambda x: datetime.fromtimestamp(x), v)))
            delta = np.diff(timestamp)
            all_td.append(delta)
    all_diff = np.concatenate(all_diff)
    all_td = np.concatenate(all_td)
    all_td_hr = np.array(list(map(lambda x: round(x.days * 24 + x.seconds / 3600, 3), all_td)))
    return all_diff.mean(), all_diff.std(), all_diff.max(), \
           round(all_td_hr.mean(),3), round(all_td_hr.std(),3), round(all_td_hr.max(), 3)

train_td_mean, train_td_std, train_td_max, train_td_hr_mean, train_td_hr_std, train_td_hr_max = normalize_td(train_data)



neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

def get_return_time(dataset):
    reoccur_dict = {}
    dataset_src, dataset_dst, dataset_t = dataset.src.numpy(), dataset.dst.numpy(), dataset.t.numpy()
    for i in range(len(dataset_src)):
        n1, n2, t = dataset_src[i], dataset_dst[i], dataset_t[i]
        key = (n1, n2)
        if key not in reoccur_dict:
            reoccur_dict[key] = [t]
        elif t == reoccur_dict[key][-1]:
            continue
        else:
            reoccur_dict[key].append(t)
    count = 0
    for _, occ in reoccur_dict.items():
        if len(occ) > 1:
            count += len(occ)-1
    print("Number of repeat events in the data : {}/{}".format(count, len(dataset_src)))
    end_time = dataset_t[-1]+1
    reoccur_time_ts = np.zeros(len(dataset_src))
    reoccur_time_hr = np.zeros(len(dataset_src))
    for idx in range(len(dataset_src)):
        n1, n2, t = dataset_src[idx], dataset_dst[idx], dataset_t[idx]
        occ = reoccur_dict[(n1,n2)]
        if len(occ) == 1 or t == occ[-1]:
            reoccur_time_ts[idx] = end_time
            reoccur_time = datetime.fromtimestamp(int(end_time)) - datetime.fromtimestamp(int(t))
            reoccur_time_hr[idx] = round((reoccur_time.days*24 + reoccur_time.seconds/3600),3)
        else:
            reoccur_time_ts[idx] = occ[occ.index(t) + 1]
            reoccur_time = datetime.fromtimestamp(int(reoccur_time_ts[idx])) - datetime.fromtimestamp(int(t))
            reoccur_time_hr[idx] = round((reoccur_time.days*24 + reoccur_time.seconds/3600),3)

    return reoccur_dict, reoccur_time_ts, reoccur_time_hr

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

class DyRepDecoder(torch.nn.Module):
    def __init__(self, embedding_dim, num_surv_samples):
        super(DyRepDecoder, self).__init__()
        self.embed_dim = embedding_dim
        self.num_surv_samples = num_surv_samples
        self.omega = Linear(in_features=2*self.embed_dim, out_features=1)
        self.psi = Parameter(0.5*torch.ones(1))
        self.alpha = Parameter(torch.rand(1))
        self.w_t = Parameter(torch.rand(1))
        self.reset_parameters()


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def forward(self, all_embeddings, assoc, src, pos_dst, neg_dst_surv,
                neg_src_surv=None, neg_dst=None, last_update=None, cur_time=None):

        z_src, z_dst = all_embeddings[assoc[src]], all_embeddings[assoc[pos_dst]]
        z_neg_dst_surv = all_embeddings[assoc[neg_dst_surv]]

        if last_update is None:
            lambda_uv = self.compute_intensity_lambda(z_src, z_dst)

            surv_u = self.compute_intensity_lambda(
                z_src.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                z_neg_dst_surv)
        else:
            last_time_pos = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[pos_dst]].view(-1,1)), dim=1).max(-1)[0]
            last_time_neg_surv = torch.cat((last_update[assoc[src]].unsqueeze(1).repeat(1,  self.num_surv_samples).view(-1,1),
                                       last_update[assoc[neg_dst_surv]].view(-1,1)), dim=1).max(-1)[0]

            ########## compute the time difference in hours
            # td_pos = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), cur_time.numpy()))) - \
            #     np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), last_time_pos.numpy())))
            #
            # td_pos_hr = torch.tensor(list(map(lambda x: round(x.days*24 + x.seconds/3600, 3), td_pos)))
            #
            # td_neg_dst = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), cur_time.unsqueeze(1).
            #                                     repeat(1, self.num_surv_samples).view(-1).numpy()))) - \
            #                   np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), last_time_neg_surv.numpy())))
            # td_neg_dst_hr = torch.tensor(list(map(lambda x: round(x.days*24 + x.seconds/3600, 3), td_neg_dst)))
            td_pos = cur_time - last_time_pos
            td_neg_dst = cur_time.unsqueeze(1).repeat(1, self.num_surv_samples).view(-1) - last_time_neg_surv

            lambda_uv = self.hawkes_intensity(z_src, z_dst, td_pos)
            surv_u = self.hawkes_intensity(
                z_src.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
                z_neg_dst_surv, td_neg_dst)

        if neg_src_surv is not None:
            z_neg_src_surv = all_embeddings[assoc[neg_src_surv]]
            surv_v = self.compute_intensity_lambda(
                z_neg_src_surv,
                z_dst.unsqueeze(1).repeat(1, self.num_surv_samples, 1).view(-1, self.embed_dim))
        else:
            surv_v = torch.tensor(0)

        loss_lambda = -torch.sum(torch.log(lambda_uv + 1e-10))
        # loss_surv = ((torch.sum(surv_u)+torch.sum(surv_v)) / self.num_surv_samples)
        loss_surv_u = torch.sum(surv_u) / self.num_surv_samples
        loss_surv_v = torch.sum(surv_v) / self.num_surv_samples

        cond_density = []
        if not self.training:
            z_neg_dst = all_embeddings[assoc[neg_dst]]
            if last_update is None:
                lambda_uv_neg = self.compute_intensity_lambda(z_src, z_neg_dst)
            else:
                last_time_neg = torch.cat((last_update[assoc[src]].view(-1,1), last_update[assoc[neg_dst]].view(-1,1)), dim=1).max(-1)[0]
                # td_neg = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), cur_time.numpy()))) - \
                #          np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), last_time_neg.numpy())))
                #
                # td_neg_hr = torch.tensor(list(map(lambda x: round(x.days * 24 + x.seconds / 3600, 3), td_neg)))

                td_neg = cur_time - last_time_neg
                lambda_uv_neg = self.hawkes_intensity(z_src, z_neg_dst, td_neg)

            s = surv_u.view(-1, self.num_surv_samples).mean(dim=-1) + surv_v.view(-1, self.num_surv_samples).mean(dim=-1)
            surv = torch.exp(-s)
            assert len(z_src) == len(surv)
            cond_pos = lambda_uv * surv
            cond_neg = lambda_uv_neg * surv
            cond_density = [cond_pos, cond_neg]
        # return loss_lambda/len(z_src), loss_surv/len(z_src)
        return loss_lambda/len(z_src), loss_surv_u/len(z_src), loss_surv_v/len(z_src), cond_density

    def hawkes_intensity(self, z_u, z_v, td, symmetric=False):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        td_norm = td / train_td_max
        # td_norm = (td - train_td_mean) / train_td_std
        if symmetric:
            g_uv = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
            g_vu = self.omega(torch.cat((z_v, z_u), dim=1)).flatten()
            g = 0.5*(g_uv + g_vu)
        else:
            g = self.omega(torch.cat((z_u, z_v), dim=1)).flatten()
        g_psi = g / (self.psi + 1e-7)
        # g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * torch.log(1 + torch.exp(g_psi))
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi) + self.alpha*torch.exp(-self.w_t*td_norm)
        return Lambda

    def compute_intensity_lambda(self, z_u, z_v):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        z_cat = torch.cat((z_u, z_v), dim=1)
        g = self.omega(z_cat).flatten()

        # g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        g_psi = g / (self.psi + 1e-7)
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi)

        return Lambda

    def g_fn(self, z1, z2):
        z_cat = torch.cat((z1, z2), dim=1)
        g = self.omega(z_cat)
        g = g.flatten()
        return g

    # compute the intensity lambda (symmetric)
    def intensity_rate_lambda(self, z_u, z_v):
        z_u = z_u.view(-1, self.embed_dim).contiguous()
        z_v = z_v.view(-1, self.embed_dim).contiguous()
        g = 0.5 * (self.g_fn(z_u, z_v) + self.g_fn(z_v, z_u))
        g_psi = g / (self.psi + 1e-7)
        # g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # to prevent overflow
        Lambda = self.psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi)
        return Lambda

memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)


num_surv_samples = 10
num_time_samples = 5

dyrep = DyRepDecoder(
    embedding_dim=embedding_dim,
    num_surv_samples=num_surv_samples
).to(device)

# link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

# optimizer = torch.optim.Adam(
#     set(memory.parameters()) | set(gnn.parameters())
#     | set(link_pred.parameters()), lr=0.0001)
optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(dyrep.parameters()), lr=0.0001)

# criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


# Get return time for val and test dataset

val_reoccur_dict, val_return_ts, val_return_hr = get_return_time(val_data)

test_reoccur_dict, test_return_ts, test_return_hr = get_return_time(test_data)

first_batch = []
def train():
    memory.train()
    gnn.train()
    dyrep.train()
    # link_pred.train()
    random_state = np.random.RandomState(12345)
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    total_loss_lambda, total_loss_surv_u, total_loss_surv_v = 0, 0, 0

    for batch_id, batch in enumerate(tqdm(train_data.seq_batches(batch_size=200), total=552)):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        ################ Sample negative destination nodes.
        # neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
        #                         dtype=torch.long, device=device)
        ################ Sample negative destination nodes all dst
        # neg_dst_surv = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0)*num_surv_samples, ),
        #                              dtype=torch.long, device=device)

        ################ Sample negative destination nodes from non-happened dst
        neg_dst_nodes = np.delete(np.arange(min_dst_idx, max_dst_idx + 1), pos_dst.numpy() - min_dst_idx)
        neg_dst_surv = torch.tensor(random_state.choice(neg_dst_nodes, size=src.size(0)*num_surv_samples,
                                                replace=len(neg_dst_nodes) < src.size(0)*num_surv_samples))

        # neg_dst_surv = []
        # for d in pos_dst.numpy():
        #     unique_dst = pos_dst.unique().numpy()
        #     nodes = np.delete(unique_dst, np.where(unique_dst==d)[0])
        #     neg_d_surv = random_state.choice(nodes, size=num_surv_samples,
        #                                         replace=len(nodes) < num_surv_samples)
        #     neg_dst_surv.extend(neg_d_surv)
        # neg_dst_surv = torch.tensor(neg_dst_surv)


        ################ Sample negative source nodes all src
        # neg_src_surv = torch.randint(min_src_idx, max_src_idx + 1, (src.size(0)*num_surv_samples, ),
        #                              dtype=torch.long, device=device)
        ################ Sample negative destination nodes from non-happened dst
        neg_src_nodes = np.delete(np.arange(min_src_idx, max_src_idx + 1), src.numpy() - min_src_idx)
        neg_src_surv = torch.tensor(random_state.choice(neg_src_nodes, size=src.size(0)*num_surv_samples,
                                                replace=len(neg_src_nodes) < src.size(0)*num_surv_samples))


        # n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        ######### only contained sampled negative
        n_id = torch.cat([src, pos_dst, neg_dst_surv, neg_src_surv]).unique()
        # n_id = torch.cat([src, pos_dst, neg_dst_surv]).unique()
        ######### include all of the negative nodes (only dst)
        # n_id=torch.cat([src, pos_dst, torch.arange(min_dst_idx, max_dst_idx+1)])

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        # loss = dyrep(z[assoc[src]], z[assoc[pos_dst]], z[assoc[neg_dst]])
        loss_lambda, loss_surv_u, loss_surv_v, _ = dyrep(z, assoc, src, pos_dst, neg_dst_surv,
                                                         neg_src_surv=neg_src_surv, last_update=None, cur_time=None)
        loss = loss_lambda + loss_surv_u + loss_surv_v

        if (batch_id) % 100 == 0:
            if batch_id==0:
                first_batch.append(loss)
            print("Batch {}, Loss {}, loss_lambda {}, loss_surv_u {}, loss_surv_v {}".format(
                batch_id+1, loss, loss_lambda, loss_surv_u, loss_surv_v))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        # torch.nn.utils.clip_grad_value_(dyrep.parameters(), 100)
        optimizer.step()
        # optimizer2.step()
        # dyrep.psi.data = torch.clamp(dyrep.psi.data, 1e-1, 1e+3)
        memory.detach()
        total_loss += float(loss) * batch.num_events
        total_loss_lambda += float(loss_lambda) * batch.num_events
        total_loss_surv_u += float(loss_surv_u) * batch.num_events
        total_loss_surv_v += float(loss_surv_v) * batch.num_events

    return total_loss / train_data.num_events, total_loss_lambda/train_data.num_events, \
           total_loss_surv_u/train_data.num_events, total_loss_surv_v/train_data.num_events


@torch.no_grad()
def test(inference_data, return_time_hr=None):
    memory.eval()
    gnn.eval()
    dyrep.eval()
    # link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.
    random_state = np.random.RandomState(12345)
    aps, aucs = [], []
    time_maes = []
    total_loss, total_maes = 0, 0
    for batch_id, batch in enumerate(tqdm(inference_data.seq_batches(batch_size=200), total=119)):
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
        #                         dtype=torch.long, device=device)

        # Negative sampling for the survival function

        # neg_dst_surv = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0)*num_surv_samples, ),
        #                              dtype=torch.long, device=device)

        neg_dst_nodes = np.delete(np.arange(min_dst_idx, max_dst_idx + 1), pos_dst.numpy() - min_dst_idx)

        neg_dst = torch.tensor(random_state.choice(neg_dst_nodes, size=src.size(0),
                                                replace=len(neg_dst_nodes) < src.size(0)))

        neg_dst_surv = torch.tensor(random_state.choice(neg_dst_nodes, size=src.size(0)*num_surv_samples,
                                                replace=len(neg_dst_nodes) < src.size(0)*num_surv_samples))

        # neg_src_surv = torch.randint(min_src_idx, max_src_idx + 1, (src.size(0)*num_surv_samples, ),
        #                              dtype=torch.long, device=device)

        neg_src_nodes = np.delete(np.arange(min_src_idx, max_src_idx + 1), src.numpy() - min_src_idx)
        neg_src_surv = torch.tensor(random_state.choice(neg_src_nodes, size=src.size(0)*num_surv_samples,
                                                replace=len(neg_src_nodes) < src.size(0)*num_surv_samples))

        all_dst = torch.arange(min_dst_idx, max_dst_idx)

        # n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # n_id = torch.cat([src, pos_dst, neg_dst_surv, neg_src_surv, neg_dst]).unique()
        # n_id = torch.cat([src, pos_dst, neg_dst_surv, neg_dst]).unique()
        ####### include all dst nodes
        n_id = torch.cat([src, all_dst, neg_src_surv]).unique()

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        assert z.size(0) > max(assoc[n_id])
        loss_lambda, loss_surv_u, loss_surv_v, cond = dyrep(z, assoc, src, pos_dst, neg_dst_surv,
                                                            neg_src_surv=neg_src_surv, neg_dst=neg_dst,
                                                            last_update=None, cur_time = None)
        # loss = dyrep(z, last_update, t, assoc, src, pos_dst, neg_src_surv, neg_dst_surv)
        pos_out, neg_out = cond[0], cond[1]
        y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        total_loss += float(loss) * batch.num_events

        # memory.update_state(src, pos_dst, t, msg)
        # neighbor_loader.insert(src, pos_dst)

        return_time_pred = []
        for src_c, pos_dst_c, t_c, msg_c in zip(src, pos_dst, t, msg):
            memory.update_state(src_c.expand(2), pos_dst_c.expand(2), t_c.expand(2), msg_c.view(1,-1).expand(2,-1))

            t_cur_date = datetime.fromtimestamp(int(t_c))
            t_prev = datetime.fromtimestamp(int(max(last_update[assoc[src_c]], last_update[assoc[pos_dst_c]])))
            td = t_cur_date - t_prev
            time_scale_hour = round((td.days * 24 + td.seconds / 3600), 3)

            embeddings_u, embeddings_v = [], []
            surv_allsamples = torch.zeros(num_time_samples)

            factor_samples = 2 * random_state.rand(num_time_samples)
            sampled_time_scale = time_scale_hour * factor_samples

            msg_src_c_store, msg_dst_c_store = copy.deepcopy(memory.msg_s_store[src_c.item()]), copy.deepcopy(memory.msg_d_store[pos_dst_c.item()])
            memory_src_c, memory_dst_c = memory.memory[src_c].clone(), memory.memory[pos_dst_c].clone()
            last_update_src_c, last_update_dst_c = memory.last_update[src_c].clone(), memory.last_update[pos_dst_c].clone()
            for n in range(1, num_time_samples+1):
                td_hours_n = sum(sampled_time_scale[:n])
                t_c_n = int((t_cur_date + timedelta(hours=td_hours_n)).timestamp())

                # neg_dst_c = torch.randint(min_dst_idx, max_dst_idx + 1, (num_surv_samples, ),
                #                 dtype=torch.long, device=device)
                neg_dst_c = torch.tensor(random_state.choice(neg_dst_nodes, size=num_surv_samples))

                # neg_src_c = torch.randint(min_src_idx, max_src_idx + 1, (num_surv_samples, ),
                #                 dtype=torch.long, device=device)
                neg_src_c = torch.tensor(random_state.choice(neg_src_nodes, size=num_surv_samples))

                n_id_c = torch.cat([src_c.expand(2), pos_dst_c.expand(2), neg_dst_c, neg_src_c]).unique()
                n_id_c, edge_index_c, e_id_c = neighbor_loader(n_id_c)
                assoc[n_id_c] = torch.arange(n_id_c.size(0), device=device)
                z_sample, last_update_sample = memory(n_id_c)
                # print((src_c,pos_dst_c, t_c), z_sample[[assoc[src_c], assoc[pos_dst_c]], :])
                embeddings_u.append(z_sample[assoc[src_c]])
                embeddings_v.append(z_sample[assoc[pos_dst_c]])
                surv_sample_u = dyrep.compute_intensity_lambda(
                    z_sample[assoc[src_c]].view(1, -1).expand(num_surv_samples, -1),
                    z_sample[assoc[neg_dst_c]])
                surv_sample_v = dyrep.compute_intensity_lambda(
                    z_sample[assoc[neg_src_c]],
                    z_sample[assoc[pos_dst_c]].view(1, -1).expand(num_surv_samples, -1))

                surv_allsamples[n-1] = torch.sum(surv_sample_u + surv_sample_v) / num_surv_samples
                memory.update_state(src_c.expand(2), pos_dst_c.expand(2), torch.tensor([t_c_n, t_c_n]), msg_c.view(1,-1).expand(2,-1))
            # TODO: update the neigh_loader one by one

            memory.msg_s_store[src_c.item()], memory.msg_d_store[pos_dst_c.item()] = copy.deepcopy(msg_src_c_store), copy.deepcopy(msg_dst_c_store)
            memory.memory[src_c], memory.memory[pos_dst_c] = memory_src_c.clone(), memory_dst_c.clone()
            memory.last_update[src_c], memory.last_update[pos_dst_c] = last_update_src_c.clone(), last_update_dst_c.clone()

            embeddings_u = torch.stack(embeddings_u, dim=0)
            embeddings_v = torch.stack(embeddings_v, dim=0)
            lambda_t_allsamples = dyrep.compute_intensity_lambda(embeddings_u, embeddings_v)
            f_samples = lambda_t_allsamples * torch.exp(-surv_allsamples)
            expectation = ((torch.from_numpy(np.cumsum(sampled_time_scale)) - train_td_hr_mean)/train_td_hr_std) * f_samples
            return_time_pred.append(expectation.sum()/num_time_samples)
        # memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        return_time_pred = torch.stack(return_time_pred).numpy()*train_td_hr_std + train_td_hr_mean
        mae = np.mean(abs(return_time_pred - return_time_hr[batch_id*200:(batch_id*200+batch.num_events)]))
        if batch_id % 20 == 0:
            print("Test Batch {}, MAE for time prediction {}, loss {}".format(batch_id+1, mae, loss))
        total_maes += mae*len(batch.src)
        time_maes.append(mae)

    # print("Finish testing, MAE for time prediction {}".format(total_maes/inference_data.num_events))

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean()), \
           total_loss/inference_data.num_events, total_maes/inference_data.num_events

all_loss, all_loss_lambda, all_loss_surv_u, all_loss_surv_v  = [], [], [], []
all_val_loss, all_test_loss = [], []
all_val_ap, all_val_auc, all_test_ap, all_test_auc = [], [], [], []
all_val_mae, all_test_mae = [], []
epochs = 20
for epoch in range(1, epochs+1): #51
    loss, loss_lambda, loss_surv_u, loss_surv_v = train()
    all_loss.append(loss)
    all_loss_lambda.append(loss_lambda)
    all_loss_surv_u.append(loss_surv_u)
    all_loss_surv_v.append(loss_surv_v)
    print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}, loss_lambda:{loss_lambda:.4f}, '
          f'loss_surv_u:{loss_surv_u:.4f}, loss_surv_v:{loss_surv_v: .4f}')
    val_ap, val_auc, val_loss, val_mae = test(val_data, val_return_hr)
    print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}, Val LOSS: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
    all_val_ap.append(val_ap)
    all_val_auc.append(val_auc)
    all_val_loss.append(val_loss)
    all_val_mae.append(val_mae)
    # test_ap, test_auc, test_loss = test(test_data)
    # print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
    # all_test_ap.append(test_ap)
    # all_test_auc.append(test_auc)
    # all_test_loss.append(test_loss)
    # print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    # print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

fig = plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(np.arange(1, epochs+1), np.array(all_loss), 'k', label='total loss')
plt.plot(np.arange(1, epochs+1), np.array(all_loss_lambda), 'r', label='loss events')
plt.plot(np.arange(1, epochs+1), np.array(all_loss_surv_u), 'b', label='loss nonevents (neg dst)')
plt.plot(np.arange(1, epochs+1), np.array(all_loss_surv_v), 'g', label='loss nonevents (neg src)')
plt.legend()
plt.title("TGN + DyRep, wiki, 50 epochs")
plt.subplot(1,2,2)
plt.plot(np.arange(1,epochs+1), np.array(first_batch), 'r')
plt.title("TGN + DyRep, loss for the first batch")
fig.savefig('tgnDyrep_wiki_oneSurv_train.png')

fig2 = plt.figure(figsize=(18, 5))
plt.subplot(1,3,1)
plt.plot(np.arange(1, epochs+1), np.array(all_val_loss), 'k', label='total loss')
plt.title("val loss")
plt.subplot(1,3,2)
plt.plot(np.arange(1, epochs+1), np.array(all_val_ap), 'r', label='total loss')
plt.title("val ap")
plt.subplot(1,3,3)
plt.plot(np.arange(1, epochs+1), np.array(all_val_mae), 'b', label='total loss')
plt.title("val mae")
fig2.savefig('tgnDyrep_wiki_oneSurv_val.png')

    # val_ap, val_auc = test(val_data, val_return_hr)
    # test_ap, test_auc = test(test_data, test_return_hr)
    # print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    # print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')