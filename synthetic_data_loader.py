import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from data_loader import EventsDataset


class SyntheticDataset(EventsDataset):

    def __init__(self, split, dataset_name, data_dir=None, link_feat=False):
        super(SyntheticDataset, self).__init__()

        self.rnd = np.random.RandomState(1111)
        self.dataset_name = dataset_name

        self.link_feat = link_feat

        graph_df = pd.read_csv('./Synthetic/ml_{}.csv'.format(dataset_name))
        graph_df = graph_df.sort_values('ts')
        test_time = np.quantile(graph_df.ts, 0.70)
        sources = graph_df.u.values
        destinations = graph_df.i.values
        event_type = graph_df.event_types.values

        # if not all_comms:
        #     visited = set()
        #     for idx, (source, des) in enumerate(zip(sources, destinations)):
        #         if (source,des) not in visited:
        #             event_type[idx]=0
        #             visited.add((source,des))

        timestamps = graph_df.ts.values
        timestamps_date = np.array(list(map(lambda x: datetime.fromtimestamp(int(x), tz=None), timestamps)))

        train_mask = timestamps<=test_time
        test_mask = timestamps>test_time

        all_events = list(zip(sources, destinations, event_type, timestamps_date))

        if split == 'train':
            self.all_events = np.array(all_events)[train_mask].tolist()
        elif split == 'test':
            self.all_events = np.array(all_events)[test_mask].tolist()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime.fromtimestamp(0)
        self.END_DATE = timestamps_date[-1]

        self.N_nodes = max(sources.max(),destinations.max())+1

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
