import platform

import numpy as np
import sys
import os
import time
import copy
import pickle
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse

from torch import autograd

from jodie_data_loader import JodieDataset
from social_data_loader import SocialEvolutionDataset
from github_data_loader import GithubDataset
from synthetic_data_loader import SyntheticDataset
from utils import *
from dyrep import DyRep
from tqdm import tqdm

def get_return_time(data_set):
    reoccur_dict = {}
    for n1,n2,r,t in data_set.all_events:
        et = 0 if r in data_set.assoc_types else 1
        key = (n1,n2,et) if n1<=n2 else (n2,n1,et)
        ts = t.timestamp()
        if key not in reoccur_dict:
            reoccur_dict[key] = [ts]
        elif ts ==  reoccur_dict[key][-1]:
            continue
        else:
            reoccur_dict[key].append(ts)
    count = 0
    for event in reoccur_dict:
        occ = reoccur_dict[event]
        if len(occ) > 1:
            count += len(occ)-1
    print("Number of repeat events in the test set: {}".format(count))
    reoccur_time_true = np.zeros(len(data_set.all_events))
    for idx, (n1,n2,r,t) in enumerate(data_set.all_events):
        et = 0 if r in data_set.assoc_types else 1
        key = (n1,n2,et) if n1<=n2 else (n2,n1,et)
        val = reoccur_dict[key]
        if len(val) == 1 or t.timestamp()==val[-1]:
            reoccur_time_true[idx] = test_loader.dataset.END_DATE.timestamp()
        else:
            reoccur_time_true[idx] = val[val.index(t.timestamp()) + 1]
    return reoccur_dict, reoccur_time_true

def mae_error(u, v, k, time_cur, expected_time, reoccur_dict, end_date):
    u, v, time_cur = u.data.cpu().numpy(), v.data.cpu().numpy(), time_cur.data.cpu().numpy()
    et = (k>0).int().data.cpu().numpy()
    batch_predict_time = []
    N = len(u)
    ae = 0
    for idx in range(N):
        key = (u[idx], v[idx], et[idx]) if u[idx] <= v[idx] else (v[idx], u[idx], et[idx])
        val = reoccur_dict[key]
        td_pred_hour = expected_time[idx]
        if len(val) == 1 or time_cur[idx]==val[-1]:
            next_ts = end_date.timestamp()
        else:
            next_ts = val[val.index(time_cur[idx])+1]
        true_td = datetime.fromtimestamp(int(next_ts))-datetime.fromtimestamp(int(time_cur[idx]))
        td_true_hour = round((true_td.days*24 + true_td.seconds/3600), 3)
        ae += abs(td_pred_hour-td_true_hour)
        batch_predict_time.append((td_pred_hour, td_true_hour))
    return ae, batch_predict_time

def MAE(expected_time_hour, batch_ts_true, t_cur):
    t_cur = t_cur.data.cpu().numpy()
    valid_idx = np.where(batch_ts_true != 0)
    t_cur_dt = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), t_cur[valid_idx])))
    batch_dt_true = np.array(list(map(lambda x: datetime.fromtimestamp(int(x)), batch_ts_true[valid_idx])))
    batch_time_true = batch_dt_true - t_cur_dt
    batch_time_hour_true = np.array(list(map(lambda td: round(td.days * 24 + td.seconds/3600, 3), batch_time_true)))
    expected_time_hour = np.array(expected_time_hour)[valid_idx]
    batch_ae = sum(abs(expected_time_hour-batch_time_hour_true))
    batch_res = list(zip(expected_time_hour, batch_time_hour_true))
    return batch_ae, batch_res

def test_time_pred(model, reoccur_dict, reoccur_time_true):
    model.eval()
    loss = 0
    losses =[ [np.Inf, 0], [np.Inf, 0] ]
    # Time slots with 10 days intervals as in the DyRep paper
    # timeslots = [t.toordinal() for t in test_loader.dataset.TEST_TIMESLOTS]

    total_ae, total_sample_num = 0, 0.000001
    # all_res = []
    test_loader.dataset.time_bar = np.zeros((test_loader.dataset.N_nodes, 1)) + test_loader.dataset.FIRST_DATE.timestamp()
    end_date = test_loader.dataset.END_DATE
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data[2] = data[2].float().to(args.device)
            data[4] = data[4].double().to(args.device)
            data[5] = data[5].double()
            batch_size = len(data[0])
            output = model(data)

            loss += (-torch.sum(torch.log(output[0]) + 1e-10) + torch.sum(output[1])).item()
            for i in range(len(losses)):
                m1 = output[i].min()
                m2 = output[i].max()
                if m1 < losses[i][0]:
                    losses[i][0] = m1
                if m2 > losses[i][1]:
                    losses[i][1] = m2
            u, v, k, time_cur = data[0], data[1], data[3], data[5]
            ########### use the event sequence to predict time
            # if batch_idx == 0:
            #     start_time = time_cur[0]
            # ae, batch_pred_res = time_prediction_error(A_pred, u, v, k, time_cur, start_time, Survival_term, reoccur_dict)
            ########### predict with repeat current event sequence with reoccur_dict
            ae, batch_pred_res = mae_error(u,v,k,time_cur,output[-1],reoccur_dict, end_date)
            ########### predict with repeat current event sequence with reoccur_time_true
            # ae, batch_pred_res = MAE(output[-1], reoccur_time_true[batch_idx*batch_size:(batch_idx+1)*batch_size], time_cur)
            ###########
            total_ae += ae
            total_sample_num += len(batch_pred_res)
            if batch_idx % 20 == 0:
                print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}'.
                      format(batch_idx + 1, len(test_loader), (total_ae / total_sample_num),
                             (loss / ((batch_idx + 1)*batch_size))))

    print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}, loss_event min/max={:.4f}/{:.4f}, '
          'loss_nonevent min/max={:.4f}/{:.4f}'.
          format(batch_idx + 1, len(test_loader), (total_ae/total_sample_num), (loss / len(test_loader.dataset)),
                 losses[0][0], losses[0][1], losses[1][0], losses[1][1],
                 len(model.Lambda_dict), time_iter / (batch_idx + 1)))
    return total_ae/total_sample_num, loss/len(test_loader.dataset)

def test(model, reoccur_dict, n_test_batches=None):
    model.eval()
    loss = 0
    losses =[ [np.Inf, 0], [np.Inf, 0] ]
    n_samples = 0
    # Time slots with 10 days intervals as in the DyRep paper
    timeslots = [t.toordinal() for t in test_loader.dataset.TEST_TIMESLOTS]
    end_date = test_set.END_DATE
    event_types = list(test_loader.dataset.event_types_num.keys()) #['comm', 'assoc']
    # sort it by k
    for event_t in test_loader.dataset.event_types_num:
        event_types[test_loader.dataset.event_types_num[event_t]] = event_t

    ## Com means the communication event type (will not change the network structure)
    event_types += ['Com']
    total_ae, total_sample_num = 0, 0.000001
    mar, hits_10 = {}, {}
    for event_t in event_types:
        mar[event_t] = []
        hits_10[event_t] = []
        for c, slot in enumerate(timeslots):
            mar[event_t].append([])
            hits_10[event_t].append([])

    start = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data[2] = data[2].float().to(args.device)
            data[4] = data[4].double().to(args.device)
            data[5] = data[5].double()
            batch_size = len(data[0])
            output = model(data)

            loss += (-torch.sum(torch.log(output[0]) + 1e-10) + torch.sum(output[1])).item()
            for i in range(len(losses)):
                m1 = output[i].min()
                m2 = output[i].max()
                if m1 < losses[i][0]:
                    losses[i][0] = m1
                if m2 > losses[i][1]:
                    losses[i][1] = m2
            n_samples += 1
            A_pred, Survival_term = output[2], output[3]
            u, v, k, time_cur = data[0], data[1], data[3], data[5]

            ae, batch_pred_res = mae_error(u, v, k, time_cur, output[-1], reoccur_dict, end_date)

            total_ae += ae
            total_sample_num += len(batch_pred_res)

            m, h = MAR(A_pred, u, v, k, Survival_term=Survival_term)
            assert len(time_cur) == len(m) == len(h) == len(k)
            for t, m, h, k_ in zip(time_cur, m, h, k):
                d = datetime.fromtimestamp(t.item()).toordinal()
                event_t = event_types[k_.item()]
                for c, slot in enumerate(timeslots):
                    if d <= slot:
                        mar[event_t][c].append(m)
                        hits_10[event_t][c].append(h)
                        if k_ > 0:
                            mar['Com'][c].append(m)
                            hits_10['Com'][c].append(h)
                        if c > 0:
                            assert slot > timeslots[c-1] and d > timeslots[c-1], (d, slot, timeslots[c-1])
                        break

            if batch_idx % 20 == 0:
                print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}'.
                      format(batch_idx + 1, len(test_loader), (total_ae / total_sample_num),
                             (loss / ((batch_idx + 1)*batch_size))))

            if n_test_batches is not None and batch_idx >= n_test_batches - 1:
                break

    time_iter = time.time() - start


    print('\nTEST batch={}/{}, time prediction MAE {}, loss={:.3f}, psi={}, loss1 min/max={:.4f}/{:.4f}, '
          'loss2 min/max={:.4f}/{:.4f}, integral time stamps={}, sec/iter={:.4f}'.
          format(batch_idx + 1, len(test_loader), (total_ae/total_sample_num), (loss / n_samples),
                 [model.psi[c].item() for c in range(len(model.psi))],
                 losses[0][0], losses[0][1], losses[1][0], losses[1][1],
                 len(model.Lambda_dict), time_iter / (batch_idx + 1)))

    # Report results for different time slots in the test set
    for c, slot in enumerate(timeslots):
        s = 'Slot {}: '.format(c)
        for event_t in event_types:
            sfx = '' if event_t == event_types[-1] else ', '
            if len(mar[event_t][c]) > 0:
                s += '{} ({} events): MAR={:.2f}+-{:.2f}, HITS_10={:.3f}+-{:.3f}'.\
                    format(event_t, len(mar[event_t][c]), np.mean(mar[event_t][c]), np.std(mar[event_t][c]),
                           np.mean(hits_10[event_t][c]), np.std(hits_10[event_t][c]))
            else:
                s += '{} (no events)'.format(event_t)
            s += sfx
        print(s)

    mar_all, hits_10_all = {}, {}
    for event_t in event_types:
        mar_all[event_t] = []
        hits_10_all[event_t] = []
        for c, slot in enumerate(timeslots):
            mar_all[event_t].extend(mar[event_t][c])
            hits_10_all[event_t].extend(hits_10[event_t][c])

    s = 'All slots: '
    for event_t in event_types:
        sfx = '' if event_t == event_types[-1] else ', '
        if len(mar_all[event_t]) > 0:
            s += '{} ({} events): MAR={:.2f}+-{:.2f}, HITS_10={:.3f}+-{:.3f}'.\
                format(event_t, len(mar_all[event_t]), np.mean(mar_all[event_t]), np.std(mar_all[event_t]),
                       np.mean(hits_10_all[event_t]), np.std(hits_10_all[event_t]))
        else:
            s += '{} (no events)'.format(event_t)
        s += sfx
    print(s)

    return mar_all, hits_10_all, loss / n_samples, total_ae/total_sample_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DyRep Model Training Parameters')

    parser.add_argument('--dataset', type=str, default='github', choices=['github', 'social', 'wikipedia', 'reddit', 'synthetic'])
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden layer dimension in DyRep')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay_step', type=str, default='10', help='number of epochs after which to reduce lr')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--all_comms', type=bool, default=False, help='assume all of the links in Jodie as communication or not')
    parser.add_argument('--include_link_feat', type=bool, default=False, help='include link features or not')
    args = parser.parse_args()
    args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    # Set seed
    np.random.seed(args.seed)
    rnd = np.random.RandomState(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.dataset=='github':
        train_set = GithubDataset('train', data_dir=args.data_dir)
        test_set = GithubDataset('test', data_dir=args.data_dir)
        initial_embeddings = np.random.randn(train_set.N_nodes, args.hidden_dim)
        A_initial = train_set.get_Adjacency()
    elif args.dataset == 'social':
        data = SocialEvolutionDataset.load_data(args.data_dir, 0.8)
        train_set = SocialEvolutionDataset(data['initial_embeddings'], data['train'], 'CloseFriend')
        test_set = SocialEvolutionDataset(data['initial_embeddings'], data['test'], 'CloseFriend',
                                          data_train=data['train'])
        initial_embeddings = data['initial_embeddings'].copy()
        A_initial = train_set.get_Adjacency()[0]
    elif args.dataset=='wikipedia':
        train_set = JodieDataset('train', dataset_name='wikipedia', all_comms=args.all_comms, data_dir=args.data_dir)
        test_set = JodieDataset('test', dataset_name='wikipedia', all_comms=args.all_comms, data_dir=args.data_dir)
        initial_embeddings = np.random.randn (train_set.N_nodes, args.hidden_dim)
        A_initial = train_set.get_Adjacency()
    elif args.dataset=='reddit':
        train_set = JodieDataset('train', dataset_name='reddit', data_dir=args.data_dir)
        test_set = JodieDataset('test', dataset_name='reddit', data_dir=args.data_dir)
        initial_embeddings = np.random.randn (train_set.N_nodes, args.hidden_dim)
        A_initial = train_set.get_Adjacency()
    elif args.dataset=='synthetic':
        train_set = SyntheticDataset('train', dataset_name='hawkes', data_dir=args.data_dir)
        test_set = SyntheticDataset('test', dataset_name='hawkes', data_dir=args.data_dir)
        initial_embeddings = np.random.randn (train_set.N_nodes, args.hidden_dim)
        A_initial = train_set.get_Adjacency()
    else:
        raise NotImplementedError(args.dataset)


    time_bar_initial = np.zeros((train_set.N_nodes, 1)) + train_set.FIRST_DATE.timestamp()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    test_reoccur_dict, test_reoccur_time_true = get_return_time(test_set)

    end_date = test_set.END_DATE

    model = DyRep(num_nodes=train_set.N_nodes,
                  hidden_dim=args.hidden_dim,
                  random_state= rnd,
                  first_date=train_set.FIRST_DATE,
                  end_datetime=end_date,
                  num_neg_samples=5,
                  num_time_samples=10,
                  device=args.device,
                  all_comms=args.all_comms).to(args.device)

    print(model)
    print('number of training parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    params_main = [param for param in model.parameters() if param.requires_grad]

    optimizer = optim.Adam(params_main, lr=args.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.5)

    for arg in vars(args):
        print(arg, getattr(args, arg))

    dt = datetime.now()
    print('start time:', dt)
    experiment_ID = '%s_%06d' % (platform.node(), dt.microsecond)
    print('experiment_ID: ', experiment_ID)

    epoch_start = 1
    batch_start = 0

    total_losses = []
    test_MAR, test_HITS10, test_loss = [], [], []
    for epoch in range(epoch_start, args.epochs + 1):
        # def reset_state(self, node_embeddings_initial, A_initial, node_degree_initial, time_bar, resetS=False):
        # Reinitialize node embeddings and adjacency matrices, but keep the model parameters intact
        model.train()
        if not isinstance(A_initial, list):
            A_initial = [A_initial]
        node_degree_initial = []
        for at, A in enumerate(A_initial):
            node_degree_initial.append(np.sum(A, axis=1))
        if len(A_initial) == 1: A_initial = A_initial[0]
        time_bar = copy.deepcopy(time_bar_initial)
        model.reset_state(node_embeddings_initial=initial_embeddings,
                          A_initial=A_initial,
                          node_degree_initial=node_degree_initial,
                          time_bar=time_bar,
                          resetS=(epoch==epoch_start))
        train_loader.dataset.time_bar = time_bar
        test_loader.dataset.time_bar = time_bar

        start = time.time()
        total_loss = 0
        for batch_idx, data_batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            data_batch[2] = data_batch[2].float().to(args.device)
            data_batch[4] = data_batch[4].double().to(args.device)
            data_batch[5] = data_batch[5].double()# no need of GPU
            if args.include_link_feat:
                data_batch[6] = data_batch[6].float().to(args.device)
            output = model(data_batch)
            losses = [-torch.sum(torch.log(output[0]) + 1e-10), torch.sum(output[1])]
            loss = torch.sum(torch.stack(losses))/args.batch_size
            loss.backward()
            # TODO: test the clip value  for model paramters
            nn.utils.clip_grad_value_(model.parameters(), 100)

            optimizer.step()
            model.psi.data = torch.clamp(model.psi.data, 1e-1, 1e+3)  # to prevent overflow in computing Lambda
            time_iter = time.time() - start
            model.z = model.z.detach()  # to reset the computational graph and avoid backpropagating second time
            model.S = model.S.detach()
            if batch_idx % 50 == 0:
                print("Training epoch {}, batch {}/{}, loss {}".format(epoch, batch_idx+1, len(train_loader), loss))
                # result = test_time_pred(model, test_reoccur_dict, test_reoccur_time_true)
                # result = test(model, test_reoccur_dict)
            total_loss += loss*args.batch_size
            scheduler.step()

        total_loss = float(total_loss)/len(train_set.all_events)
        total_losses.append(total_loss)
        print("Training epoch {}/{}, time per batch {}, total loss {}".format(epoch, args.epochs + 1, time_iter/float(batch_idx+1), total_loss))

        print("Testing Start")
        result = test_time_pred(model, test_reoccur_dict, test_reoccur_time_true)
        # result = test(model, test_reoccur_dict)
        print("Test end")

        # result = test(model, n_test_batches=None)
        # test_MAR.append(np.mean(result[0]['Com']))
        # test_HITS10.append(np.mean(result[1]['Com']))
        # test_loss.append(result[2])
        # print("Testing results: MAR {}, HITS10 {}, test_loss {}".format(test_MAR[-1], test_HITS10[-1], test_loss[-1]))

    print('end time:', datetime.now())