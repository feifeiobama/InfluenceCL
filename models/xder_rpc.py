# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from datasets import get_dataset
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')

    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--m', type=float, default=0.3)


    return parser


def dsimplex(num_classes=10):
    def simplex_coordinates2(m):
        # add the credit
        x = np.zeros([m, m + 1])
        for j in range(0, m):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(float(1 + m))) / float(m)

        for i in range(0, m):
            x[i, m] = a

        #  Adjust coordinates so the centroid is at zero.
        c = np.zeros(m)
        for i in range(0, m):
            s = 0.0
            for j in range(0, m + 1):
                s = s + x[i, j]
            c[i] = s / float(m + 1)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] - c[i]

        #  Scale so each column has norm 1. UNIT NORMALIZED
        s = 0.0
        for i in range(0, m):
            s = s + x[i, 0] ** 2
        s = np.sqrt(s)

        for j in range(0, m + 1):
            for i in range(0, m):
                x[i, j] = x[i, j] / s

        return x

    feat_dim = num_classes - 1
    ds = simplex_coordinates2(feat_dim)
    return ds

class XDerRPC(ContinualModel):
    NAME = 'xder_rpc'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(XDerRPC, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.tasks = get_dataset(args).N_TASKS
        self.task = 0
        self.update_counter = torch.zeros(self.args.buffer_size).to(self.device)
        self.pernicehead = torch.from_numpy(dsimplex(self.cpt * self.tasks)).float().to(self.device)

        if not hasattr(self.args, 'start_from'):
            self.args.start_from=0

    def forward(self, x):
        x = self.net(x)[:, :-1]
        if x.dtype != self.pernicehead.dtype:
            self.pernicehead = self.pernicehead.type(x.dtype)
        x = x @ self.pernicehead
        return x
    def end_task(self, dataset):

        tng = self.training
        self.train()

        if self.args.start_from is None or self.task >= self.args.start_from:
            # Reduce Memory Buffer
            if self.task > 0:
                examples_per_class = self.args.buffer_size // ((self.task + 1) * self.cpt)
                buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
                self.buffer.empty()
                for tl in buf_lab.unique():
                    idx = tl == buf_lab
                    ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                    first = min(ex.shape[0], examples_per_class)
                    self.buffer.add_data(
                        examples=ex[:first],
                        labels = lab[:first],
                        logits=log[:first],
                        task_labels=tasklab[:first]
                    )

            # Add new task data
            examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
            examples_per_class = examples_last_task // self.cpt
            ce = torch.tensor([examples_per_class] * self.cpt).int()
            ce[torch.randperm(self.cpt)[:examples_last_task - (examples_per_class * self.cpt)]] += 1

            with torch.no_grad():
                with bn_track_stats(self, False):
                    if self.args.start_from is None or self.args.start_from <= self.task:
                        for data in dataset.train_loader:
                            inputs, labels, not_aug_inputs = data
                            inputs = inputs.to(self.device)
                            not_aug_inputs = not_aug_inputs.to(self.device)
                            outputs = self(inputs)
                            if all(ce == 0):
                                break

                            # Update past logits
                            if self.task > 0:
                                outputs = self.update_logits(outputs, outputs, labels, 0, self.task)

                            flags = torch.zeros(len(inputs)).bool()
                            for j in range(len(flags)):
                                if ce[labels[j] % self.cpt] > 0:
                                    flags[j] = True
                                    ce[labels[j] % self.cpt] -= 1

                            self.buffer.add_data(examples=not_aug_inputs[flags],
                                                labels=labels[flags],
                                                logits=outputs.data[flags],
                                                task_labels=(torch.ones(len(not_aug_inputs)) *
                                                            (self.task))[flags])

                    # Update future past logits
                    buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(self.buffer.buffer_size,
                        transform=self.transform, return_index=True)


                    buf_outputs = []
                    while len(buf_inputs):
                        buf_outputs.append(self(buf_inputs[:self.args.batch_size]))
                        buf_inputs = buf_inputs[self.args.batch_size:]
                    buf_outputs = torch.cat(buf_outputs)

                    chosen = (buf_labels // self.cpt) < self.task

                    if chosen.any():
                        to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.task, self.tasks - self.task)
                        self.buffer.logits[buf_idx[chosen],:] = to_transplant.to(self.buffer.device)
                        self.buffer.task_labels[buf_idx[chosen]] = self.task

        self.task += 1
        self.update_counter = torch.zeros(self.args.buffer_size).to(self.device)

        self.train(tng)

    def update_logits(self, old, new, gt, task_start, n_tasks=1):

        transplant = new[:, task_start*self.cpt:(task_start+n_tasks)*self.cpt]

        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.args.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1,self.cpt * n_tasks)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1,self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, task_start*self.cpt:(task_start+n_tasks)*self.cpt] = transplant

        return old

    def observe(self, inputs, labels, not_aug_inputs):


        self.opt.zero_grad()

        outputs = self(inputs).float()

        # Present head
        loss_stream = self.loss(outputs[:,self.task*self.cpt:(self.task+1)*self.cpt], labels % self.cpt)

        loss_der, loss_derpp = torch.tensor(0.), torch.tensor(0.)
        if not self.buffer.is_empty():
            # Distillation Replay Loss (all heads)
            buf_idx1, buf_inputs1, buf_labels1, buf_logits1, buf_tl1 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_outputs1 = self(buf_inputs1).float()

            buf_logits1 = buf_logits1.type(buf_outputs1.dtype)
            mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
            loss_der = self.args.alpha * mse.mean()

            # Label Replay Loss (past heads)
            buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            buf_outputs2 = self(buf_inputs2).float()

            buf_ce = self.loss(buf_outputs2[:, :(self.task)*self.cpt], buf_labels2)
            loss_derpp = self.args.beta * buf_ce

            # Merge Batches & Remove Duplicates
            buf_idx = torch.cat([buf_idx1, buf_idx2])
            buf_inputs = torch.cat([buf_inputs1, buf_inputs2])
            buf_labels = torch.cat([buf_labels1, buf_labels2])
            buf_logits = torch.cat([buf_logits1, buf_logits2])
            buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
            buf_tl = torch.cat([buf_tl1, buf_tl2])
            eyey = torch.eye(self.buffer.buffer_size).to(self.device)[buf_idx]
            umask = (eyey * eyey.cumsum(0)).sum(1) < 2

            buf_idx = buf_idx[umask]
            buf_inputs = buf_inputs[umask]
            buf_labels = buf_labels[umask]
            buf_logits = buf_logits[umask]
            buf_outputs = buf_outputs[umask]
            buf_tl = buf_tl[umask]

            # Update Future Past Logits
            with torch.no_grad():
                chosen = (buf_labels // self.cpt) < self.task
                self.update_counter[buf_idx[chosen]] += 1
                c = chosen.clone()
                chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                if chosen.any():
                    assert self.task > 0
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.task, self.tasks - self.task)
                    self.buffer.logits[buf_idx[chosen],:] = to_transplant.to(self.buffer.device)
                    self.buffer.task_labels[buf_idx[chosen]] = self.task


        # Past Logits Constraint
        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
        if self.task > 0:
            chead = F.softmax(outputs[:, :(self.task+1)*self.cpt], 1)

            good_head = chead[:,self.task*self.cpt:(self.task+1)*self.cpt]
            bad_head  = chead[:,:self.cpt*self.task]

            loss_constr = bad_head.max(1)[0].detach() + self.args.m - good_head.max(1)[0]

            mask = loss_constr > 0

            if (mask).any():
                loss_constr_past = self.args.eta * loss_constr[mask].mean()


        # Future Logits Constraint
        loss_constr_futu = torch.tensor(0.)
        if self.task < self.tasks - 1:
            bad_head = outputs[:,(self.task+1)*self.cpt:]
            good_head = outputs[:,self.task*self.cpt:(self.task+1)*self.cpt]

            if not self.buffer.is_empty():
                buf_tlgt = buf_labels // self.cpt
                bad_head = torch.cat([bad_head, buf_outputs[:,(self.task+1)*self.cpt:]])
                good_head  = torch.cat([good_head, torch.stack(buf_outputs.split(self.cpt, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

            loss_constr = bad_head.max(1)[0] + self.args.m - good_head.max(1)[0]

            mask = loss_constr > 0
            if (mask).any():
                loss_constr_futu = self.args.eta * loss_constr[mask].mean()

        loss = loss_stream + loss_der + loss_derpp + loss_constr_futu + loss_constr_past

        loss.backward()
        self.opt.step()

        return loss.item()
