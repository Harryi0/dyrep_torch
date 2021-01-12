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
import copy

from datetime import datetime, timedelta
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

    def forward(self, z_src, z_dst, z_neg_src, z_neg_dst):

        lambda_uv = self.compute_intensity_lambda(z_src, z_dst)

        surv_u = self.compute_intensity_lambda(
            z_src.unsqueeze(1).repeat(1,  self.num_surv_samples, 1).view(-1, self.embed_dim),
            z_neg_dst)

        surv_v = self.compute_intensity_lambda(
            z_neg_src,
            z_dst.unsqueeze(1).repeat(1, self.num_surv_samples, 1).view(-1, self.embed_dim))

        loss = -torch.sum(torch.log(lambda_uv) + 1e-10) + (torch.sum(surv_u)/self.num_surv_samples) + \
               (torch.sum(surv_v)/self.num_surv_samples)

        return loss / len(z_src)

    def compute_intensity_lambda(self, z_u, z_v):
        z_u = z_u.view(-1, self.embed_dim)
        z_v = z_v.view(-1, self.embed_dim)
        z_cat = torch.cat((z_u, z_v), dim=1)
        g = self.omega(z_cat).flatten()

        g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # avoid overflow
        # Lambda = self.psi * torch.log(1 + torch.exp(g_psi))
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
        g_psi = torch.clamp(g / (self.psi + 1e-7), -75, 75)  # to prevent overflow
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


num_surv_samples = 5
num_time_samples = 5

dyrep = DyRepDecoder(
    embedding_dim=embedding_dim,
    num_surv_samples=num_surv_samples
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

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


def train():
    memory.train()
    gnn.train()
    dyrep.train()
    # link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch_id, batch in enumerate(tqdm(train_data.seq_batches(batch_size=200))):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        # neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
        #                         dtype=torch.long, device=device)
        # Sample negative destination nodes， num_surv_samples for each node in the batch
        neg_dst_surv = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0)*num_surv_samples, ),
                                     dtype=torch.long, device=device)

        # Sample negative source nodes， num_surv_samples for each node in the batch
        neg_src_surv = torch.randint(min_src_idx, max_src_idx + 1, (src.size(0)*num_surv_samples, ),
                                     dtype=torch.long, device=device)

        # n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id = torch.cat([src, pos_dst, neg_src_surv, neg_dst_surv]).unique()
        # n_id = torch.cat([src, pos_dst, neg_dst_surv]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        # loss = dyrep(z[assoc[src]], z[assoc[pos_dst]], z[assoc[neg_dst]])

        loss = dyrep(z[assoc[src]], z[assoc[pos_dst]], z[assoc[neg_src_surv]], z[assoc[neg_dst_surv]])
        # loss = dyrep(z[assoc[src]], z[assoc[pos_dst]], z[assoc[neg_dst_surv]])

        if (batch_id) % 100 == 0:
            print("Batch {}, Loss {}".format(batch_id+1, loss))

        # pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        # neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        # loss = criterion(pos_out, torch.ones_like(pos_out))
        # loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        torch.nn.utils.clip_grad_value_(dyrep.parameters(), 100)
        optimizer.step()
        dyrep.psi.data = torch.clamp(dyrep.psi.data, 1e-1, 1e+3)
        memory.detach()
        total_loss += float(loss) * batch.num_events
        # if batch_id >=20:
        #     break

    return total_loss / train_data.num_events


@torch.no_grad()
def test(inference_data, return_time_hr):
    memory.eval()
    gnn.eval()
    dyrep.eval()
    # link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.
    random_state = np.random.RandomState(12345)
    aps, aucs = [], []
    time_maes = []
    total_loss, total_maes = 0, 0
    for batch_id, batch in enumerate(tqdm(inference_data.seq_batches(batch_size=200))):
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
        #                         dtype=torch.long, device=device)

        # Negative sampling for the survival function
        neg_dst_surv = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0)*num_surv_samples, ),
                                     dtype=torch.long, device=device)
        neg_src_surv = torch.randint(min_src_idx, max_src_idx + 1, (src.size(0)*num_surv_samples, ),
                                     dtype=torch.long, device=device)

        # n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id = torch.cat([src, pos_dst, neg_dst_surv, neg_src_surv]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        loss = dyrep(z[assoc[src]], z[assoc[pos_dst]], z[assoc[neg_src_surv]], z[assoc[neg_dst_surv]])
        total_loss += float(loss) * batch.num_events
        return_time_pred = []

        # making itme prediction  for each node in the batch
        for src_c, pos_dst_c, t_c, msg_c in zip(src, pos_dst, t, msg):
            # just update the current node to the memory
            memory.update_state(src_c.expand(2), pos_dst_c.expand(2), t_c.expand(2), msg_c.view(1,-1).expand(2,-1))

            t_cur_date = datetime.fromtimestamp(int(t_c))
            # Take the most recent last update time in the node pair
            t_prev = datetime.fromtimestamp(int(max(last_update[assoc[src_c]], last_update[assoc[pos_dst_c]])))
            # The time difference between current time and most recent update time would be a base for the future time sampling
            td = t_cur_date - t_prev
            time_scale_hour = round((td.days * 24 + td.seconds / 3600), 3)
            embeddings_u, embeddings_v = [], []
            surv_allsamples = torch.zeros(num_time_samples)
            # random generate factor [0,2] for the time sampling
            factor_samples = 2 * random_state.rand(num_time_samples)
            sampled_time_scale = time_scale_hour * factor_samples
            # make back up for the msg store, memoruy and
            msg_src_c_store, msg_dst_c_store = copy.deepcopy(memory.msg_s_store[src_c.item()]), copy.deepcopy(memory.msg_d_store[pos_dst_c.item()])
            memory_src_c, memory_dst_c = memory.memory[src_c].clone(), memory.memory[pos_dst_c].clone()
            last_update_src_c, last_update_dst_c = memory.last_update[src_c].clone(), memory.last_update[pos_dst_c].clone()

            for n in range(1, num_time_samples+1):
                td_hours_n = sum(sampled_time_scale[:n])
                t_c_n = int((t_cur_date + timedelta(hours=td_hours_n)).timestamp())

                neg_dst_c = torch.randint(min_dst_idx, max_dst_idx + 1, (num_surv_samples, ),
                                dtype=torch.long, device=device)
                neg_src_c = torch.randint(min_src_idx, max_src_idx + 1, (num_surv_samples, ),
                                dtype=torch.long, device=device)
                n_id_c = torch.cat([src_c.expand(2), pos_dst_c.expand(2), neg_dst_c, neg_src_c]).unique()
                n_id_c, edge_index_c, e_id_c = neighbor_loader(n_id_c)
                assoc[n_id_c] = torch.arange(n_id_c.size(0), device=device)
                z_sample, last_update_sample = memory(n_id_c)
                embeddings_u.append(z_sample[assoc[src_c]])
                embeddings_v.append(z_sample[assoc[pos_dst_c]])
                surv_sample_u = dyrep.intensity_rate_lambda(
                    z_sample[assoc[src_c]].view(1, -1).expand(num_surv_samples, -1),
                    z_sample[assoc[neg_dst_c]])
                surv_sample_v = dyrep.intensity_rate_lambda(
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
            lambda_t_allsamples = dyrep.intensity_rate_lambda(embeddings_u, embeddings_v)
            f_samples = lambda_t_allsamples * surv_allsamples
            expectation = torch.from_numpy(np.cumsum(sampled_time_scale)) * f_samples
            return_time_pred.append(expectation.sum())
        neighbor_loader.insert(src, pos_dst)
        return_time_pred = torch.stack(return_time_pred).numpy()
        mae = np.mean(abs(return_time_pred - return_time_hr[batch_id*200:(batch_id*200+batch.num_events)]))
        if batch_id % 20 == 0:
            print("Test Batch {}, MAE for time prediction {}, loss {}".format(batch_id+1, mae, loss))
        total_maes += mae*len(batch.src)
        time_maes.append(mae)

    print("Finish testing, MAE for time prediction {}".format(total_maes/inference_data.num_events))



    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


for epoch in range(1, 51):
    loss = train()
    print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_data, val_return_hr)
    # test_ap, test_auc = test(test_data, test_return_hr)
    # print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    # print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')