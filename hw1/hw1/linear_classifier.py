import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader 

from .losses import ClassifierLoss

class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal distribution with zero mean and the given std.
        

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.normal(mean=torch.zeros(n_features, n_classes), std=weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = x@self.weights
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = 1/len(y)*torch.sum(y_pred - y == 0)
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=1000,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print(self.weights.shape)

        print("Training\n", end="")
        for epoch_idx in range(max_epochs):
            print(f'epoch {epoch_idx}')
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            loss_train = 0
            accuracy_train = 0
            print('train')
            for i, batch_train in enumerate(dl_train):
                print(f'batch {i}')
                # every batch has batch_size samples and lables
                # dl train in this case is 8000 samples i.e. 8 batches of 1000 val has 2000
                
                y_pred, x_scores = self.predict(batch_train[0])
                
                batch_loss = loss_fn(batch_train[0], batch_train[1], x_scores) + weight_decay/2*torch.sum(self.weights**2)
                loss_train = loss_train + batch_loss
                
                grad = loss_fn.grad() + weight_decay*self.weights
                print(f'grad shape: {grad.shape}')
                self.weights = self.weights - learn_rate* grad

                accuracy_train = accuracy_train + self.evaluate_accuracy(batch_train[1], y_pred)    
            
            loss_train = loss_train/len(dl_train)
            train_res[0].append(accuracy_train)
            train_res[1].append(loss_train)
            
            print('val')
            loss_val = 0
            accuracy_val = 0
            for j, batch_val in enumerate(dl_valid):
                print(f'batch {j}')
                y_pred, x_scores = self.predict(batch_val[0])
                
                batch_loss = loss_fn(batch_val[0], batch_val[1], x_scores) + weight_decay/2*torch.sum(self.weights**2)
                loss_val = loss_val + batch_loss

                accuracy_val = accuracy_val + self.evaluate_accuracy(batch_val[1], y_pred)

            loss_val = loss_val/len(dl_valid)
            valid_res[1].append(loss_val)
            valid_res[0].append(accuracy_val)
            

            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.01
    hp['learn_rate'] = 0.1
    hp['weight_decay'] = 0.1
    # ========================

    return hp
