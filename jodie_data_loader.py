import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timezone
from data_loader import EventsDataset


class JodieDataset(EventsDataset):

    def __init__(self, split, dataset_name, all_comms=False, data_dir=None, link_feat=False):
        super(JodieDataset, self).__init__()

        self.rnd = np.random.RandomState(1111)
        self.dataset_name = dataset_name

        self.link_feat = link_feat

        ### take a subset of 15000
        # allgraph_df = pd.read_csv('./Jodie/ml_{}.csv'.format(dataset_name))
        # allgraph_df = allgraph_df.sort_values('ts')
        # graph_df = allgraph_df[:15000]

        graph_df = pd.read_csv('./Jodie/ml_{}.csv'.format(dataset_name))
        graph_df = graph_df.sort_values('ts')
        graph_df['event_types'] = 1
        val_time = np.quantile(graph_df.ts, 0.70)
        # val_time = np.quantile(graph_df.ts, 0.85)
        test_time = np.quantile(graph_df.ts, 0.85)
        sources = graph_df.u.values-1
        destinations = graph_df.i.values-1
        shift = min(destinations)-max(sources)-1
        destinations = destinations-shift
        event_type = graph_df.event_types.values

        self.min_src_idx, self.max_src_idx = min(sources), max(sources)
        self.min_dst_idx, self.max_dst_idx = min(destinations), max(destinations)

        if not all_comms:
            visited = set()
            for idx, (source, des) in enumerate(zip(sources, destinations)):
                if (source,des) not in visited:
                    event_type[idx]=0
                    visited.add((source,des))

        link_features = np.load('./Jodie/ml_{}.npy'.format(dataset_name))[1:]
        timestamps = graph_df.ts.values
        # timestamps_date = graph_df.ts.apply(lambda x: datetime.fromtimestamp((int(x)), tz=None))
        timestamps_date = np.array(list(map(lambda x: datetime.fromtimestamp(int(x), tz=None), timestamps)))

        train_mask = timestamps<=val_time
        # test_mask = timestamps>val_time
        val_mask = (timestamps>val_time) & (timestamps<=test_time)
        test_mask = timestamps>test_time

        if self.link_feat:
            all_events = list(zip(sources , destinations, event_type, timestamps_date, list(link_features)))
        else:
            all_events = list(zip(sources, destinations, event_type, timestamps_date))

        if split == 'train':
            self.all_events = np.array(all_events)[train_mask].tolist()
        elif split == 'test':
            self.all_events = np.array(all_events)[val_mask].tolist()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime.fromtimestamp(0)
        # self.END_DATE = timestamps_date[-1]
        self.END_DATE = self.all_events[-1][3]

        self.N_nodes = len(np.unique(sources)) + len(np.unique(destinations))

        assert self.N_nodes == max(destinations)+1

        self.n_events = len(self.all_events)

        self.event_types_num = {0: 0, 1:1}

        self.assoc_types = [0]

        self.A_initial = np.zeros((self.N_nodes, self.N_nodes))

        # random_source = self.rnd.choice(np.unique(sources), size=500, replace=False)
        # random_des =self.rnd.choice(np.unique(destinations), size=500, replace=False)
        #
        # for i, j  in zip(random_source, random_des):
        #     self.A_initial[i,j] = 1
        #     self.A_initial[j,i] = 1

        print('\nA_initial', np.sum(self.A_initial))


    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: Github has only one relation type (FollowEvent), so multirelations are ignored')
        return self.A_initial
