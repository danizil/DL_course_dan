import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders

class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
    
        sample_list = []
        label_list = []
        for (i, batch) in enumerate(dl_train):
            # if i ==0:
            #     print(batch[0])
            sample_list.append(batch[0])
            label_list.append(batch[1])
        x_train = torch.cat(sample_list)
        y_train = torch.cat(label_list)
        n_classes = torch.max(y_train).item() + 1
        
        # # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.
        # ====== YOUR CODE: ======
        k_argmins_mat = torch.topk(dist_matrix, self.k, dim=0, largest=False).indices
        classes = self.y_train[k_argmins_mat]
        y_pred = torch.mode(classes, dim = 0).values
        # ========================
        
        
        # for i in range(n_test):
        #     # TODO:
        #     #  - Find indices of k-nearest neighbors of test sample i
        #     #  - Set y_pred[i] to the most common class among them
        #     #  - Don't use an explicit loop.
        #     
        #     raise NotImplementedError()
        #     

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    # ====== YOUR CODE: ======
    x1x2 = x1@x2.T
    x1_sq = torch.diag(x1@x1.T)
    x2_sq = torch.diag(x2@x2.T)
    dists = torch.sqrt(-2*x1x2 + x1_sq[:, None] + x2_sq[None, :])
    
    
    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    accuracy = torch.sum((y - y_pred) == 0)/len(y)


    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []
    num_samples = len(ds_train)
    indices = torch.arange(num_samples)
    folds_indices = list(torch.split(indices, num_samples//num_folds))
    
    for i, k in enumerate(k_choices):
        print(f'k index {i} (k={k})\n\n')
        model = KNNClassifier(k)
        
        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        val_scores = []
        for j in range(num_folds):
            # print(f'fold {j} is validation')
            train_inds = torch.cat(folds_indices[:j] + folds_indices[j+1:])
            val_inds = folds_indices[j]
            
            dl_train = DataLoader(ds_train, 1024, sampler=dataloaders.IndexSampler(ds_train, train_inds))
            dl_val = DataLoader(ds_train, len(val_inds), sampler=dataloaders.IndexSampler(ds_train, val_inds))
            
            x_val , y_val = dataloader_utils.flatten(dl_val)
            model.train(dl_train)
            y_pred = model.predict(x_val)
            val_scores.append(accuracy(y_val, y_pred))
            
        accuracies.append(val_scores)
        # print(f'\naccuracies for k={k}: \n {val_scores}\n')
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
