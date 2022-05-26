import abc
from audioop import mul
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        assert delta>0
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted=torch.tensor([])):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        M = x_scores - x_scores.gather(1, y[:, None]) + self.delta
        L_is = torch.clone(M)
        L_is[L_is<0] = 0
        # COMMENT: we remove the length of the vector y times 
        # delta from the loss because delta is added to every correct element which is now zero
        loss = 1/len(y)*(torch.sum(L_is) - self.delta*len(y))
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['hinged_margin'] = L_is
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['num_classes'] = x_scores.shape[1]
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        L_is = self.grad_ctx['hinged_margin']
        x = self.grad_ctx['x'] 
        y = self.grad_ctx['y']
        num_classes = self.grad_ctx['num_classes']

        # G = L_is
        # G[G<0] = 0
        # G[G]

        NCD = x.view(x.shape[0], 1, x.shape[1]).repeat(1,num_classes, 1)
        margin_mask = L_is > 0
        NCD = NCD*margin_mask[:, :, None]
        
        # L_is[L_is==1] = 0
        multiply_yi = -(torch.sum(L_is, dim=1) - self.delta)
        mask_yi = torch.ones_like(L_is)
        mask_yi[list(range(len(y))), y] = multiply_yi # V
        NCD = NCD*mask_yi[:, :, None]
        
        grad = 1/len(y)*torch.sum(NCD, dim=0).T 
        
        # ========================

        return grad
