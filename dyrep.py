import numpy as np
import torch
from datetime import datetime, timedelta
from torch.nn import Linear, ModuleList, Parameter


class DyRep(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim, random_state, first_date, end_datetime, num_neg_samples= 5, num_time_samples = 10,
                 device='cpu', all_comms=False):
        super(DyRep, self).__init__()

        self.batch_update = True
        self.all_comms = all_comms
        self.include_link_features = False

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.first_date = first_date
        self.end_datetime = end_datetime
        self.num_neg_samples = num_neg_samples
        self.device = device
        self.num_time_samples = num_time_samples
        self.n_assoc_types = 1
        # TODO: TB we bring bias term to the linear layer by using Linear (set bias=False to exempt or directly use parameter)
        if not self.include_link_features:
            self.omega = ModuleList([Linear(in_features=2*hidden_dim, out_features=1),
                                     Linear(in_features=2*hidden_dim, out_features=1)])
        else:
            self.omega = ModuleList([Linear(in_features=2*hidden_dim+172, out_features=1),
                                     Linear(in_features=2*hidden_dim+172, out_features=1)])
        self.psi = Parameter(0.5*torch.ones(2)) # type=2: assoc + comm

        self.W_h = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_struct = Linear(in_features=hidden_dim*self.n_assoc_types, out_features=hidden_dim)
        self.W_rec = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_t = Linear(4,hidden_dim) # [days, hours, minutes, seconds]

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def reset_state(self, node_embeddings_initial, A_initial, node_degree_initial, time_bar, resetS=False):
        z = np.pad(node_embeddings_initial, ((0, 0), (0, self.hidden_dim - node_embeddings_initial.shape[1])),'constant')
        z = torch.from_numpy(z).float().to(self.device)
        A = torch.from_numpy(A_initial).float().to(self.device)
        if len(A.shape) == 2:
            A = A.unsqueeze(2)
        self.register_buffer('z', z)
        self.register_buffer('A', A)
        self.node_degree_global = node_degree_initial
        self.time_bar = time_bar

        ## TODO: Current implementation, initialize S for each epoch
        self.initialize_S_from_A()

        assert torch.sum(torch.isnan(A)) == 0, (torch.sum(torch.isnan(A)), A)

        self.Lambda_dict = torch.zeros(5000, device=self.device)
        self.time_keys = []

    def initialize_S_from_A(self):
        S = self.A.new_zeros((self.num_nodes, self.num_nodes, self.n_assoc_types))
        for at in range(self.n_assoc_types):
            D = torch.sum(self.A[:,:,at], dim=1)
            for v in torch.nonzero(D, as_tuple=False):
                u = torch.nonzero(self.A[v,:,at].squeeze(), as_tuple=False)
                S[v,u,at] = 1. / D[v]
        self.S = S
        # Check that values in each row of S add up to 1
        for rel in range(self.n_assoc_types):
            S = self.S[:, :, rel]
            assert torch.sum(S[self.A[:, :, rel] == 0]) < 1e-5, torch.sum(S[self.A[:, :, rel] == 0])

    def forward(self, data):
        # TODO: change the order and change variable names with the dataloader
        u, v, time_diff, event_types, t_bar, t = data[:6]

        batch_size = len(u)

        u_all,  v_all = u.data.cpu().numpy(), v.data.cpu().numpy()
        A_pred, surv, lambda_pred = None, None, None
        if not self.training:
            A_pred = self.A.new_zeros((batch_size, self.num_nodes, self.num_nodes))
            surv = self.A.new_zeros((batch_size, self.num_nodes, self.num_nodes))

        time_mean = torch.from_numpy(np.array([0, 0, 0, 0])).float().to(self.device).view(1, 1, 4)
        time_sd = torch.from_numpy(np.array([50, 7, 15, 15])).float().to(self.device).view(1, 1, 4)
        time_diff = (time_diff - time_mean) / time_sd

        # TODO: implement the batch update version
        lambda_uv,  lambda_uv_neg = [], []

        # for batch update
        batch_embeddings_u, batch_embeddings_v, batch_embeddings_u_neg, batch_embeddings_v_neg = [], [], [], []

        node_degrees = []
        z_all = []

        expected_time = []

        # update_node_degrees = []

        compare_embeddings_u_neg, compare_embeddings_v_neg = [], []
        for it in range(batch_size):
            u_it, v_it, et_it, td_it = u_all[it], v_all[it], event_types[it], time_diff[it]

            ### TODO: remove iterate the number of assoc types (assume it to always=1)

            ### TODO: [optimize for Jodie] Buffer the z in a list before writing it to self.z might could improve the speed?
            z_prev = self.z if it == 0 else z_all[it - 1]

            ## 1. compute intensity lambda based on the most recent node embedding

            if self.batch_update:
                batch_embeddings_u.append(z_prev[u_it])
                batch_embeddings_v.append(z_prev[v_it])
            else:
                lambda_uv_it = self.compute_intensity_lambda(z_prev[u_it], z_prev[v_it], et_it)
                lambda_uv.append(lambda_uv_it)

            ## 2. compute new node embeddings
            z_new = self.update_node_embedding(z_prev, u_it, v_it, td_it)
            # if self.batch_update: node_degrees.append(node_degree)
            assert torch.sum(torch.isnan(z_new)) == 0, (torch.sum(torch.isnan(z_new)), z_new, it)

            # update_node_degrees.append(update_node_degree)

            if not self.batch_update:
                ## 3. update S and A
                self.update_A_S(u_it, v_it, et_it, lambda_uv_it)
                ### update the global node degree
                for j in [u_it, v_it]:
                    for at in range(self.n_assoc_types):
                        # self code
                        self.node_degree_global[at][j] = torch.sum(self.A[j, :, at]>0).item()
                        # self.node_degree_global[at][j] = node_degree[j][at]

            ## 4. compute lambda for sampled events that do not happen -> to compute survival probability in loss
            # uv_others = self.random_state.choice(np.delete(np.arange(self.num_nodes), [u_it, v_it]),
            #                        size=self.num_neg_samples * 2, replace=False)
            # for q in range(self.num_neg_samples):
            #     assert u_it != uv_others[q], (u_it, uv_others[q])
            #     assert v_it != uv_others[self.num_neg_samples + q], (v_it, uv_others[self.num_neg_samples + q])
            #     if self.batch_update:
            #         batch_embeddings_u_neg.extend([z_prev[u_it], z_prev[uv_others[self.num_neg_samples + q]]])
            #         batch_embeddings_v_neg.extend([z_prev[uv_others[q]], z_prev[v_it]])

            # #######self code
            u_all_node, v_all_node = np.unique(u_all), np.unique(v_all)
            u_it_idx, v_it_idx = np.where(u_all_node==u_it), np.where(v_all_node==v_it)

            batch_nodes = np.union1d(np.delete(u_all_node, u_it_idx), np.delete(v_all_node, v_it_idx))
            batch_uv_neg = self.random_state.choice(batch_nodes, size=self.num_neg_samples * 2,
                                                    replace=len(batch_nodes)<2*self.num_neg_samples)
            batch_u_neg, batch_v_neg = batch_uv_neg[self.num_neg_samples:], batch_uv_neg[:self.num_neg_samples]
            # batch_embeddings_u_neg.append(torch.cat((z_prev[u_it].expand(self.num_neg_samples, -1),
            #                                          z_prev[batch_u_neg]), dim=0))
            # batch_embeddings_v_neg.append(torch.cat([z_prev[batch_v_neg], z_prev[v_it].expand(self.num_neg_samples, -1)], dim=0))

            # uv_neg = self.random_state.choice(np.delete(np.arange(self.num_nodes), [u_it, v_it]),
            #                                      size=self.num_neg_samples * 2, replace=False)

            for q in range(self.num_neg_samples):
                batch_embeddings_u_neg.extend([z_prev[u_it], z_prev[batch_u_neg[q]]])
                batch_embeddings_v_neg.extend([z_prev[batch_v_neg[q]], z_prev[v_it]])

            # batch_embeddings_u_neg.append(torch.cat((z_prev[u_it].expand(self.num_neg_samples, -1),
            #                                          z_prev[u_neg]), dim=0))
            # batch_embeddings_v_neg.append(torch.cat([z_prev[v_neg], z_prev[v_it].expand(self.num_neg_samples, -1)], dim=0))

            ## 5. Compute  conditional density for all possible pairs
            with torch.no_grad():
                z_uv_it = torch.cat((z_prev[u_it].detach().unsqueeze(0).expand(self.num_nodes,-1),
                           z_prev[v_it].detach().unsqueeze(0).expand(self.num_nodes, -1)), dim=0)
                # two type of events: assoc + comm
                lambda_uv_pred = self.compute_intensity_lambda(z_uv_it, z_prev.detach().repeat(2,1), et_it.repeat(len(z_uv_it))).detach()
                if not self.training:
                    A_pred[it, u_it, :] = lambda_uv_pred[:self.num_nodes]
                    A_pred[it, v_it, :] = lambda_uv_pred[self.num_nodes:]
                    assert torch.sum(torch.isnan(A_pred[it])) == 0, (it, torch.sum(torch.isnan(A_pred[it])))
                    s_u_v = self.compute_cond_density(u_it, v_it, t_bar[it])
                    surv[it, [u_it, v_it], :] = s_u_v
                time_key = int(t[it])
                idx = np.delete(np.arange(self.num_nodes), [u_it, v_it])
                idx = np.concatenate((idx, idx+self.num_nodes))
                #### if total length reach the limit, remove the oldest one
                # TODO: Rename the sequence variable and set the length as a parameter (why 5000)
                if len(self.time_keys) >= len(self.Lambda_dict):
                    time_keys = np.array(self.time_keys)
                    time_keys[:-1] = time_keys[1:]
                    self.time_keys = list(time_keys[:-1])
                    self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                    self.Lambda_dict[-1] = 0
                self.Lambda_dict[[len(self.time_keys)]] = lambda_uv_pred[idx].sum().detach()
                self.time_keys.append(time_key)
                # ###############For time prediction
                if not self.training:
                    # stored_S = self.S.clone()
                    t_cur_date = datetime.fromtimestamp(int(t[it]))
                    # Use the cur and most recent time
                    t_prev = datetime.fromtimestamp(int(max(t_bar[it][u_it], t_bar[it][v_it])))
                    td = t_cur_date - t_prev
                    time_scale_hour = round((td.days*24 + td.seconds/3600),3)
                    # delta = np.array([td.days, td.seconds // 3600, (td.seconds // 60) % 60, td.seconds % 60],
                    #                  np.float)
                    # delta = np.tile(delta, (2,1))
                    # assert delta.shape==(2,4)
                    # delta = torch.from_numpy(delta).float().to(self.device)
                    prev_embedding = z_new.clone()
                    # expectation = 0
                    lambda_t_allsamples = []
                    embeddings_u, embeddings_v = [], []
                    surv_allsamples = prev_embedding.new_zeros(self.num_time_samples)
                    factor_samples = 2*self.random_state.rand(self.num_time_samples)
                    sampled_time_scale = time_scale_hour*factor_samples

                    for n in range(1, self.num_time_samples+1):
                        # lambda_t_sample = self.compute_intensity_lambda(prev_embedding[u_it], prev_embedding[v_it],et_it)
                        # lambda_t_allsamples.append(lambda_t_sample)
                        embeddings_u.append(prev_embedding[u_it])
                        embeddings_v.append(prev_embedding[v_it])

                        td_sample = timedelta(hours=sampled_time_scale[n-1])
                        delta_sample = np.array([td_sample.days, td_sample.seconds // 3600,
                                                 (td_sample.seconds // 60) % 60, td_sample.seconds % 60],np.float)
                        delta_sample = torch.from_numpy(np.tile(delta_sample, (2, 1))).float().to(self.device)

                        new_embedding = self.update_node_embedding(prev_embedding, u_it, v_it, delta_sample)

                        # self.update_A_S(u_it, v_it, et_it, lambda_t_sample)

                        # u_neg = self.random_state.choice(np.delete(u_all_node, u_it_idx), size=self.num_neg_samples,
                        #                                  replace=(len(u_all_node) - 1 < self.num_neg_samples))
                        #
                        # v_neg = self.random_state.choice(np.delete(v_all_node, v_it_idx), size=self.num_neg_samples,
                        #                                  replace=(len(v_all_node) - 1 < self.num_neg_samples))

                        batch_uv_neg_sample = self.random_state.choice(batch_nodes, size=self.num_neg_samples * 2,
                                                                replace=len(batch_nodes) < 2 * self.num_neg_samples)
                        u_neg_sample = batch_uv_neg_sample[self.num_neg_samples:]
                        v_neg_sample = batch_uv_neg_sample[:self.num_neg_samples]
                        embeddings_u_neg = torch.cat((prev_embedding[u_it].view(1,-1).expand(self.num_neg_samples,-1),
                                                        prev_embedding[u_neg_sample]),dim=0)
                        embeddings_v_neg = torch.cat([prev_embedding[v_neg_sample],
                                                      prev_embedding[v_it].view(1,-1).expand(self.num_neg_samples,-1)],dim=0)

                        surv_sample = sum(self.compute_intensity_lambda(embeddings_u_neg, embeddings_v_neg, torch.zeros(len(embeddings_u_neg))))+\
                            sum(self.compute_intensity_lambda(embeddings_u_neg, embeddings_v_neg, torch.ones(len(embeddings_u_neg))))

                        surv_allsamples[n-1] = surv_sample/self.num_neg_samples

                        prev_embedding = new_embedding

                    embeddings_u = torch.stack(embeddings_u,dim=0)
                    embeddings_v = torch.stack(embeddings_v,dim=0)
                    lambda_t_allsamples = self.compute_intensity_lambda(embeddings_u,embeddings_v,
                                                                        torch.zeros(self.num_time_samples)+et_it)
                    # lambda_t_allsamples = torch.stack(lambda_t_allsamples)
                    f_samples = lambda_t_allsamples*surv_allsamples
                    expectation = torch.from_numpy(np.cumsum(sampled_time_scale))*f_samples
                    expectation = expectation.sum()

                    expected_time.append(expectation/self.num_time_samples)
                    # self.S = stored_S.clone()

            ## 6. Update the embedding z
            z_all.append(z_new)
        self.z = z_new

        #### batch update for all events' intensity

        if self.batch_update:
            batch_embeddings_u = torch.stack(batch_embeddings_u, dim=0)
            batch_embeddings_v = torch.stack(batch_embeddings_v, dim=0)
            lambda_uv = self.compute_intensity_lambda(batch_embeddings_u, batch_embeddings_v, event_types)

            for i,k in enumerate(event_types):
                u_it, v_it = u_all[i], v_all[i]
                self.update_A_S(u_it, v_it, k, lambda_uv[i].item())
                for j in [u_it, v_it]:
                    for at in range(self.n_assoc_types):
                        self.node_degree_global[at][j] = torch.sum(self.A[j, :, at]>0).item()
        else:
            lambda_uv = torch.cat(lambda_uv, dim=0)


        # batch_embeddings_u_neg = torch.stack(batch_embeddings_u_neg, dim=0)
        # batch_embeddings_v_neg = torch.stack(batch_embeddings_v_neg, dim=0)
        # non_events = len(batch_embeddings_u_neg)
        # lambda_uv_neg = torch.zeros(non_events * 2, device=self.device)
        # non_idx = None
        # empty_t = torch.zeros(non_events, dtype=torch.long)
        # types_lst = torch.arange(2)
        # for k in types_lst:
        #     if non_idx is None:
        #         non_idx = np.arange(non_events)
        #     else:
        #         non_idx += non_events
        #     lambda_uv_neg[non_idx] = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg, empty_t + k)

        # for i,k in enumerate(event_types):
        #     u_it, v_it = u_all[i], v_all[i]
        #     self.update_A_S(u_it, v_it, k, lambda_uv[i].item())

        # batch update for all non events' intesnity
        # ##### self code
        # batch_embeddings_u_neg = torch.cat(batch_embeddings_u_neg, dim=0)
        # batch_embeddings_v_neg = torch.cat(batch_embeddings_v_neg, dim=0)
        batch_embeddings_u_neg = torch.stack(batch_embeddings_u_neg, dim=0)
        batch_embeddings_v_neg = torch.stack(batch_embeddings_v_neg, dim=0)
        neg_events_len = len(batch_embeddings_u_neg)
        lambda_uv_neg = torch.zeros(neg_events_len * 2, device=self.device)
        lambda_uv_neg[:neg_events_len] = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg,
                                                                       torch.zeros(neg_events_len))
        lambda_uv_neg[neg_events_len:] = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg,
                                                                       torch.ones(neg_events_len))
        # lambda_uv_neg_0 = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg, torch.tensor(0))
        # lambda_uv_neg_1 = self.compute_intensity_lambda(batch_embeddings_u_neg, batch_embeddings_v_neg, torch.tensor(1))
        # lambda_uv_neg = (lambda_uv_neg_0 + lambda_uv_neg_1)/self.num_neg_samples


        # lambda_uv_neg = torch.cat(lambda_uv_neg, dim=0) / self.num_neg_samples

        return lambda_uv, lambda_uv_neg / self.num_neg_samples, A_pred, surv, expected_time
        # return lambda_uv, lambda_uv_neg, A_pred, surv, expected_time

    def compute_intensity_lambda(self, z_u, z_v, et_uv):
        z_u = z_u.view(-1, self.hidden_dim)
        z_v = z_v.view(-1, self.hidden_dim)
        z_cat = torch.cat((z_u, z_v), dim=1)
        et = (et_uv>0).long()
        g = z_cat.new_zeros(len(z_cat))
        # Total two types of events
        for k in range(2):
            idx = (et==k)
            if torch.sum(idx)>0:
                g[idx] = self.omega[k](z_cat).flatten()[idx]

        psi = self.psi[et]
        g_psi = torch.clamp(g/(psi + 1e-7), -75, 75) # avoid overflow
        Lambda = psi * torch.log(1 + torch.exp(g_psi))
        return Lambda

    def update_node_embedding(self, prev_embedding, node_u, node_v, td):
        z_new = prev_embedding.clone()
        h_u_struct = prev_embedding.new_zeros((2, self.hidden_dim, self.n_assoc_types))# 2 -> update embedding for both u & v
        for cnt, (v,u) in enumerate([(node_u,  node_v), (node_v, node_u)]):
            for at in range(self.n_assoc_types):
                u_nb = self.A[u, :, at] > 0
                num_u_nb = torch.sum(u_nb).item()
                if num_u_nb > 0:
                    h_i_bar = self.W_h(prev_embedding[u_nb]).view(num_u_nb, self.hidden_dim)
                    q_ui = torch.exp(self.S[u, u_nb, at])
                    q_ui = q_ui / (torch.sum(q_ui) + 1e-7)
                    h_u_struct[cnt, :, at] = torch.max(torch.sigmoid(q_ui.view(-1,1)*h_i_bar), dim=0)[0]
        z_new[[node_u, node_v]] = torch.sigmoid(self.W_struct(h_u_struct.view(2, self.hidden_dim*self.n_assoc_types)) + \
                                  self.W_rec(prev_embedding[[node_u, node_v]]) + \
                                  self.W_t(td).view(2, self.hidden_dim))
        return z_new

    def update_A_S(self, u_it, v_it, et_it, lambda_uv_t):
        if self.all_comms:
            self.A[u_it, v_it, 0] = self.A[v_it, u_it, 0] = 1
        else:
            if et_it <= 0:
                self.A[u_it, v_it, np.abs(et_it)] = self.A[v_it, u_it, np.abs(et_it)] = 1
        A = self.A
        indices = torch.arange(self.num_nodes, device=self.device)
        for k in range(self.n_assoc_types):
            if (et_it>0) and (A[u_it, v_it, k]==0):
                continue
            else:
                for j,i in [(u_it,v_it), (v_it, u_it)]:
                    y = self.S[j, :, k]
                    # TODO: check if this work (not use the node degree when compute embedding)
                    degree_j = torch.sum(A[j,:,k] > 0).item()
                    b = 0 if degree_j==0 else 1/(float(degree_j) + 1e-7)
                    if et_it>0 and A[j,i,k]==1:
                        y[i] = b + lambda_uv_t
                    elif k==0 and A[j,i,k]==1:
                        degree_j_bar = self.node_degree_global[k][j]
                        b_prime = 0 if degree_j_bar==0 else 1./(float(degree_j_bar) + 1e-7)
                        x = b_prime - b
                        y[i] = b + lambda_uv_t
                        w_idx = (y!=0) & (indices != int(i))
                        # w_idx[int(i)] = False
                        y[w_idx] = y[w_idx]-x
                    y /= (torch.sum(y)+ 1e-7)
                    self.S[j,:,k] = y

    def compute_cond_density(self, u, v, time_bar):
        N = self.num_nodes
        s_uv = self.Lambda_dict.new_zeros((2, N))
        # TODO: why divide normalize by the length of Lambda_dict
        Lambda_sum = torch.cumsum(self.Lambda_dict.flip(0), 0).flip(0) / len(self.Lambda_dict)
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]

        indices = []
        l_indices = []
        t_bar_min = torch.min(time_bar[[u, v]]).item()
        if t_bar_min < time_keys_min:
            start_ind_min = 0
        elif t_bar_min > time_keys_max:
            # it means t_bar will always be larger, so there is no history for these nodes
            return s_uv
        else:
            start_ind_min = self.time_keys.index(int(t_bar_min))

        # Most recent time between
        max_pairs = torch.max(torch.cat((time_bar[[u, v]].view(1, 2).expand(N, -1).t().contiguous().view(2 * N, 1),
                                         time_bar.repeat(2, 1)), dim=1), dim=1)[0].view(2, N).long().data.cpu().numpy()  # 2,N

        # compute cond density for all pairs of u and some i, then of v and some i
        ############### ???
        for c, j in enumerate([u, v]):  # range(i + 1, N):
            for i in range(N):
                if i == j:
                    continue
                # most recent timestamp of either u or v
                t_bar = max_pairs[c, i]

                if t_bar < time_keys_min:
                    start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
                elif t_bar > time_keys_max:
                    continue  # it means t_bar is current event, so there is no history for this pair of nodes
                else:
                    # t_bar is somewhere in between time_keys_min and time_keys_min
                    start_ind = self.time_keys.index(t_bar, start_ind_min)

                indices.append((c, i))
                l_indices.append(start_ind)

        indices = np.array(indices)
        l_indices = np.array(l_indices)
        s_uv[indices[:, 0], indices[:, 1]] = Lambda_sum[l_indices]

        return s_uv