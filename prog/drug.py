from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import (InMemoryDataset, Data, Dataset, DataLoader)
import torch as t
from rdkit import Chem
import pickle as pkl
import random
import csv, os, sys
# import hickle as hkl
import numpy as np
import scipy.sparse as sp
import sys

from torch_geometric.utils import (dense_to_sparse, to_undirected, add_self_loops, remove_self_loops)


class EntryDataset(InMemoryDataset):
    def __init__(self, root_folder, transform=None, pre_transform=None,
                 pre_filter=None, filename='data'):
        os.makedirs(os.path.join(root_folder, 'processed'), exist_ok=True)
        super(EntryDataset, self).__init__(root_folder, transform, pre_transform, pre_filter)
        self.filename = filename

    #         if os.path.exists(self.processed_paths[0]):
    #             print('loading processed data...')
    #             tmp = t.load(self.processed_paths[0])
    #             if len(tmp) == 3:
    #                 self.data, self.slices,self.entryIDs  = tmp
    #             elif len(tmp) == 4:
    #                 self.data, self.slices,self.entryIDs,self.unpIDs  = tmp

    @property
    def processed_file_names(self, ):
        return '{}.pt'.format(self.filename)

    def add_self_loops(self, ):
        def add_self_loops_func(data):
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            return data

        data_list = [add_self_loops_func(data) for data in self]

        data, slices = self.collate(data_list)

        self.data = data
        self.slices = slices
        self.__data_list__ = data_list

    def to_fixed_atom_features(self, max_atom=100, max_atom_feature=75):

        features = []
        for graph in self:
            tmp = np.zeros([max_atom, max_atom_feature])
            feat = graph.x.detach().cpu().numpy()

            if len(feat.shape) == 1: feat = feat.reshape(-1, 1)
            use_num_atom = min(feat.shape[0], max_atom)
            use_num_atom_feat = min(feat.shape[1], max_atom_feature)
            tmp[:use_num_atom, :use_num_atom_feat] = feat[:use_num_atom, :use_num_atom_feat]
            features.append(tmp)
        return np.array(features)

    def drug_process(self, drug_df, flag_add_self_loops=False, default_dim_features=75, default_dim_nodes=50):
        import deepchem as dc
        from rdkit import Chem
        from tqdm import tqdm
        assert np.all(np.in1d(['drugID', 'SMILES'], drug_df.columns.values))
        self.entryIDs = drug_df.drugID.values

        mols_list = list(map(Chem.MolFromSmiles, drug_df.SMILES))  # some SMILES maybe are failed to parse
        featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        deepchem_list = featurizer.featurize(mols_list)

        data_list = []
        for convMol in tqdm(deepchem_list):
            # print(convMol)
            if isinstance(convMol, np.ndarray):
                print('all zeros')
                feat_mat = np.zeros((default_dim_nodes, default_dim_features))
                num_nodes = feat_mat.shape[0]
                edges = t.from_numpy(np.array([[], []])).long()


            else:
                feat_mat = convMol.get_atom_features()  # .atom_features
                degree_list = convMol.deg_list
                adj_list = convMol.get_adjacency_list()  # canon_adj_list

                num_nodes = feat_mat.shape[0]
                out_indexs = np.concatenate([[idx, ] * len(val) for idx, val in enumerate(adj_list)])
                in_indexs = np.concatenate(adj_list)
                edges = np.array([out_indexs, in_indexs])
                edges = to_undirected(t.from_numpy(edges).long(), num_nodes)

            if flag_add_self_loops:
                edges = add_self_loops(edges, num_nodes=num_nodes)[0]

            data_list.append(Data(x=t.from_numpy(feat_mat).float(), edge_index=edges, ))

        data, slices = self.collate(data_list)
        t.save((data, slices, self.entryIDs), self.processed_paths[0])
        self.data, self.slices, self.entryIDs = t.load(self.processed_paths[0])

    def drug_process_with_ConvMol_and_MolGraphConvFeaturizer(self,
                                                             drug_df, flag_add_self_loops=False,
                                                             default_dim_features=78, default_dim_nodes=50):
        import deepchem as dc
        from rdkit import Chem
        from tqdm import tqdm
        assert np.all(np.in1d(['drugID', 'SMILES'], drug_df.columns.values))
        self.entryIDs = drug_df.drugID.values

        mols_list = list(map(Chem.MolFromSmiles, drug_df.SMILES))  # some SMILES maybe are failed to parse
        featurizer = dc.feat.graph_features.ConvMolFeaturizer(use_chirality=True, )
        deepchem_list = featurizer.featurize(mols_list, )
        featurizer2 = dc.feat.MolGraphConvFeaturizer(use_edges=True,
                                                     use_chirality=True, use_partial_charge=True)
        deepchem_list2 = featurizer2.featurize(mols_list)

        data_list = []
        for convMol, MolGraphConv in tqdm(zip(deepchem_list, deepchem_list2)):

            if isinstance(convMol, np.ndarray):
                convMol_success_flag = False
                feat_mat = np.zeros((default_dim_nodes, default_dim_features))
                num_nodes = feat_mat.shape[0]
                edges = np.array([[], []])
                edges_attr = np.array([])


            else:
                # convMol
                convMol_success_flag = True
                feat_mat = convMol.get_atom_features()  # .atom_features
                degree_list = convMol.deg_list
                adj_list = convMol.get_adjacency_list()  # canon_adj_list

                num_nodes = feat_mat.shape[0]
                out_indexs = np.concatenate([[idx, ] * len(val) for idx, val in enumerate(adj_list)])
                in_indexs = np.concatenate(adj_list)
                edges = np.array([out_indexs, in_indexs])
                edges = to_undirected(t.from_numpy(edges).long(), num_nodes).detach().cpu().numpy()
                edges_attr = np.array([])

            if isinstance(MolGraphConv, np.ndarray):
                MolGraphConv_success_flag = False
                feat_mat2 = np.zeros((default_dim_nodes, default_dim_features))
                num_nodes = feat_mat2.shape[0]
                edges2 = np.array([[], []])
                edges_attr2 = np.array([])
            else:
                MolGraphConv_success_flag = True
                feat_mat2 = MolGraphConv.node_features  # .atom_features
                num_nodes = feat_mat2.shape[0]
                edges_attr2 = MolGraphConv.edge_features
                edges2 = MolGraphConv.edge_index

            if (convMol_success_flag == True) and (MolGraphConv_success_flag == True):
                edges_attr = edges_attr2
                edges = edges2
                # feat_mat = feat_mat
            elif (convMol_success_flag == False) and (MolGraphConv_success_flag == True):
                edges_attr = edges_attr2
                edges = edges2
                feat_mat = feat_mat  # 不能是feat_mat2,长度不一样

            if flag_add_self_loops:
                edges = add_self_loops(edges, num_nodes=num_nodes)[0]

            data_list.append(Data(x=t.from_numpy(feat_mat).float(),
                                  edge_index=t.from_numpy(edges).long(),
                                  edge_attr=t.from_numpy(edges_attr).float()))

        data, slices = self.collate(data_list)
        t.save((data, slices, self.entryIDs), self.processed_paths[0])
        self.data, self.slices, self.entryIDs = t.load(self.processed_paths[0])

    def drug_process_with_MolGraphConvFeaturizer(self,
                                                 drug_df, flag_add_self_loops=False,
                                                 default_dim_features=33, default_dim_nodes=50):
        import deepchem as dc
        from rdkit import Chem
        from tqdm import tqdm
        assert np.all(np.in1d(['drugID', 'SMILES'], drug_df.columns.values))
        self.entryIDs = drug_df.drugID.values

        mols_list = list(map(Chem.MolFromSmiles, drug_df.SMILES))  # some SMILES maybe are failed to parse
        # featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        deepchem_list = featurizer.featurize(mols_list)

        data_list = []
        for convMol in tqdm(deepchem_list):
            # print(convMol)
            if isinstance(convMol, np.ndarray):
                feat_mat = np.zeros((default_dim_nodes, default_dim_features))
                num_nodes = feat_mat.shape[0]
                edges = np.array([[], []])
                edges_attr = np.array([])


            else:
                feat_mat = convMol.node_features  # .atom_features
                num_nodes = feat_mat.shape[0]
                edges_attr = convMol.edge_features
                edges = convMol.edge_index

            if flag_add_self_loops:
                edges = add_self_loops(edges, num_nodes=num_nodes)[0]

            data_list.append(Data(x=t.from_numpy(feat_mat).float(),
                                  edge_index=t.from_numpy(edges).long(),
                                  edge_attr=t.from_numpy(edges_attr).float()))

        data, slices = self.collate(data_list)
        t.save((data, slices, self.entryIDs), self.processed_paths[0])
        self.data, self.slices, self.entryIDs = t.load(self.processed_paths[0])
