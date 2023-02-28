# Reference: https://github.com/zalanborsos/bilevel_coresets
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
import copy
from scipy.sparse.linalg import cg, LinearOperator


def cross_entropy_ntk(K, alpha, y, weights=1., lmbda=0.):
    loss = torch.mean(F.cross_entropy(torch.matmul(K, alpha), y.long(), reduction='none') * weights)
    if lmbda > 0:
        loss += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss


def flat_grad(grad, reshape=False, detach=False):
    # reshape implies detach
    if reshape:
        return torch.cat([p.detach().reshape(-1) for p in grad])
    if detach:
        return torch.cat([p.detach().view(-1) for p in grad])
    return torch.cat([p.view(-1) for p in grad])


def calc_hvp(loss, params, v):
    dl_p = flat_grad(grad(loss, params, create_graph=True, retain_graph=True))
    return flat_grad(grad(dl_p, params, grad_outputs=v, retain_graph=True), reshape=True)


def calc_hvp_v(inner_loss, outer_loss, params, s):
    dl_p = flat_grad(grad(outer_loss, params, create_graph=True))
    dl_p_in = flat_grad(grad(inner_loss, params, create_graph=True))
    hvp = flat_grad(grad(dl_p_in, params, grad_outputs=s, create_graph=True))
    return hvp, dl_p


class InfluenceNTK:
    """"
    Influence Functions with Neural Tangent Kernel

    Args:
        loss_fn (function): loss function
        out_dim (int): output dimension
        max_it (int): maximum number of iterations for solving the ioptimization
        lr (float): learning rate of the optimizer (L-BFGS)
        max_conj_grad_it (int): number of conjugate gradient steps in the approximate Hessian-vector products
        candidate_batch_size (int): number of candidate points considered in each selection step
        div_tol (float): divergence tolerance threshild for the optimization problem
    """

    def __init__(self, out_dim=10, max_it=300, lr=0.25, max_conj_grad_it=50, div_tol=10):
        self.loss_fn = cross_entropy_ntk
        self.out_dim = out_dim
        self.max_it = max_it
        self.lr = lr
        self.max_conj_grad_it = max_conj_grad_it
        self.div_tol = div_tol

    def solve_representer_proxy(self, K, y, lmbda, weights=1.):
        m = K.shape[1]

        # initialize the representer coefficients
        alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
        alpha.data *= 0.01

        loss = np.inf
        while loss > self.div_tol:

            def closure():
                optimizer.zero_grad()
                loss = self.loss_fn(K, alpha, y, weights=weights, lmbda=lmbda)
                loss.backward()
                return loss

            optimizer = torch.optim.LBFGS([alpha], lr=self.lr, max_iter=self.max_it)

            optimizer.step(closure)
            loss = self.loss_fn(K, alpha, y, weights=weights, lmbda=lmbda)
            if loss > self.div_tol:
                # reinitialize upon divergence
                print("Warning: opt diverged, try setting lower learning rate.")
                alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
                alpha.data *= 0.01

        return alpha

    def calc_ihvp(self, loss, params, v):
        # no necessity to refactor this to perform cg in pytorch
        op = LinearOperator((len(v), len(v)),
                            matvec=lambda x: calc_hvp(loss, params, torch.from_numpy(x).float()).numpy())
        return torch.from_numpy(cg(op, v.numpy(), maxiter=self.max_conj_grad_it)[0]).float()

    def calc_influences(self, inner_loss, outer_loss, weights, params):
        dg_dalpha = flat_grad(grad(outer_loss, params), detach=True)
        ihvp = self.calc_ihvp(inner_loss, params, dg_dalpha)
        dg_dtheta = flat_grad(grad(inner_loss, params, create_graph=True))
        weights_grad = - flat_grad(grad(dg_dtheta, weights, grad_outputs=ihvp), reshape=True)
        return weights_grad, dg_dalpha, ihvp

    def calc_second_influences(self, K, alpha, y, v, s, weights, lmbda, mu=0., outer_weights=1.):
        if (outer_weights - weights).sum() > 0:
            inner_loss = self.loss_fn(K, alpha, y, outer_weights - weights, lmbda)
            outer_loss = self.loss_fn(K, alpha, y, outer_weights - weights)
            hvp, dl_p = calc_hvp_v(inner_loss, outer_loss, alpha, s)
        else:
            inner_loss = self.loss_fn(K, alpha, y, weights - outer_weights, lmbda)
            outer_loss = self.loss_fn(K, alpha, y, weights - outer_weights)
            hvp, dl_p = calc_hvp_v(inner_loss, outer_loss, alpha, s)
        stat = torch.norm(dl_p - mu * hvp)
        influences_ext = flat_grad(grad(stat, weights), reshape=True)
        return influences_ext, stat.detach().item()

    def select(self, X, y, m, kernel_fn_np, lmbda=1e-4, mu=0., nu=0., inc_weight=1.):
        n = X.shape[0]
        chosen_indexes = np.arange(n)

        X = X.numpy()
        K = torch.from_numpy(kernel_fn_np(X, X)).float()

        outer_weights = torch.ones([n])
        outer_weights[m:] = inc_weight
        weights = torch.ones([n], requires_grad=True)
        weights.data[m:] = inc_weight

        alpha = self.solve_representer_proxy(K, y, lmbda, weights=outer_weights).requires_grad_()

        inner_loss = self.loss_fn(K, alpha, y, weights, lmbda)
        outer_loss = self.loss_fn(K, alpha, y, outer_weights)
        influences, v, s = self.calc_influences(inner_loss, outer_loss, weights, alpha)
        del inner_loss, outer_loss

        if nu == 0:
            delete_indexes = influences.argsort()[m:]
            chosen_indexes = np.delete(chosen_indexes, delete_indexes)
        else:
            weights.data[m:] = 1
            for _ in range(n, m, -1):
                second_influences, stat = self.calc_second_influences(K, alpha, y, v, s, weights, lmbda, mu=mu, outer_weights=outer_weights)
                delete_index = (influences + nu * second_influences)[chosen_indexes].argsort()[-1]
                weights.data[chosen_indexes[delete_index]] = 0
                chosen_indexes = np.delete(chosen_indexes, delete_index)

        if nu == 0:
            return chosen_indexes, influences.mean(), influences.std()
        else:
            return chosen_indexes, influences.std(), second_influences.std(), stat
