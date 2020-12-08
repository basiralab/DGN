from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
import numpy as np
import os
#We used 35813 (part of the Fibonacci Sequence) as the seed
np.random.seed(35813)

def create_better_simulated(N_Subjects, N_ROIs):
    """
        Simulated dataset distributions are inspired from real measurements
        so this function creates better dataset for demo.
        However, number of views are hardcoded.
    """
    features =  np.triu_indices(N_ROIs)[0].shape[0]
    view1 = np.random.normal(0.1,0.069, (N_Subjects, features))
    view1 = view1.clip(min = 0)
    view1 = np.array([antiVectorize(v, N_ROIs) for v in view1])
    
    view2 = np.random.normal(0.72,0.5, (N_Subjects, features))
    view2 = view2.clip(min = 0)
    view2 = np.array([antiVectorize(v, N_ROIs) for v in view2])
    
    view3 = np.random.normal(0.32,0.20, (N_Subjects, features))
    view3 = view3.clip(min = 0)
    view3 = np.array([antiVectorize(v, N_ROIs) for v in view3])
    
    view4 = np.random.normal(0.03,0.015, (N_Subjects, features))
    view4 = view4.clip(min = 0)
    view4 = np.array([antiVectorize(v, N_ROIs) for v in view4])
    
    return np.stack((view1, view2, view3, view4), axis = 3)

def simulate_dataset(N_Subjects, N_ROIs, N_views):
    """
        Creates random dataset
        Args:
            N_Subjects: number of subjects
            N_ROIs: number of region of interests
            N_views: number of views
        Return:
            dataset: random dataset with shape [N_Subjects, N_ROIs, N_ROIs, N_views]
    """
    features =  np.triu_indices(N_ROIs)[0].shape[0]
    views = []
    for _ in range(N_views):
        view = np.random.uniform(0.1,2, (N_Subjects, features))
        
        view = np.array([antiVectorize(v, N_ROIs) for v in view])
        views.append(view)
    return np.stack(views, axis = 3)

#Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

#Antivectorize given vector (this gives an asymmetric adjacency matrix)
#def antiVectorize(vec, m):
#    M = np.zeros((m,m))
#     M[np.triu_indices(m)] = vec
#     M[np.tril_indices(m)] = vec
#     M[np.diag_indices(m)] = 0
#     return M

#Antivectorize given vector (this gives a symmetric adjacency matrix)
def antiVectorize(vec, m):
    """
    #Old Code
    M = np.zeros((m,m))
    M[np.triu_indices(m)] = vec
    M[np.tril_indices(m)] = vec
    M[np.diag_indices(m)] = 0
    """
    # Correct:
    M = np.zeros((m,m))
    M[np.tril_indices(m,k=-1)] = vec
    M= M.transpose()
    M[np.tril_indices(m,k=-1)] = vec
    return M

#CV splits and mean-std calculation for the loss function
def preprocess_data_array(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    train_channel_means = np.mean(X_train, axis=(0,1,2))
    train_channel_std =   np.std(X_train, axis=(0,1,2))
    return X_train, X_test, train_channel_means, train_channel_std


#Create data objects  for the DGN
#https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
def cast_data(array_of_tensors, subject_type = None, flat_mask = None):
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    dataset = []
    for mat in array_of_tensors:
            #Allocate numpy arrays 
            edge_index = np.zeros((2, N_ROI * N_ROI))
            edge_attr = np.zeros((N_ROI * N_ROI,CHANNELS))
            x = np.zeros((N_ROI, 1))
            y = np.zeros((1,))
            
            counter = 0
            for i in range(N_ROI):
                for j in range(N_ROI):
                    edge_index[:, counter] = [i, j]
                    edge_attr[counter, :] = mat[i, j]
                    counter += 1
    
            #Fill node feature matrix (no features every node is 1)
            for i in range(N_ROI):
                x[i,0] = 1
                
            #Get graph labels
            y[0] = None
            
            if flat_mask is not None:
                edge_index_masked = []
                edge_attr_masked = []
                for i,val in enumerate(flat_mask):
                    if val == 1:
                        edge_index_masked.append(edge_index[:,i])
                        edge_attr_masked.append(edge_attr[i,:])
                edge_index = np.array(edge_index_masked).T
                edge_attr = edge_attr_masked
            

            edge_index = torch.tensor(edge_index, dtype = torch.long)
            edge_attr = torch.tensor(edge_attr, dtype = torch.float)
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            con_mat = torch.tensor(mat, dtype=torch.float)
            data = Data(x = x, edge_index=edge_index, edge_attr=edge_attr, con_mat = con_mat,  y=y, label = subject_type)
            dataset.append(data)
    return dataset
