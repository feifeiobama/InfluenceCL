import torch
from utils.ifs_buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
from utils.ntk_generator import get_kernel_fn
from utils.influence_ntk import InfluenceNTK


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--sel_epoch', type=int, nargs='+', default=[49],
                        help='Epoch for sample selection')
    parser.add_argument('--mu', type=float, default=0.5,
                        help='Probability of already-in-coreset case.')
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Weight for second-order influence functions.')
    parser.add_argument('--lmbda', type=float, default=1e-3,
                        help='Regularization coefficient for NTK.')
    parser.add_argument('--norig', action='store_true',
                        help='Not store original image in the buffer')
    return parser


class SOIF(ContinualModel):
    NAME = 'soif'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(SOIF, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.kernel_fn = get_kernel_fn(backbone)
        self.ifs = InfluenceNTK()

    def begin_task(self, dataset):
        self.task += 1
        self.epoch = 0
        self.ifs.out_dim = dataset.N_CLASSES_PER_TASK * self.task

    def end_epoch(self, dataset):
        self.epoch += 1

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            indexes, buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        if self.epoch in self.args.sel_epoch:
            inputs = inputs if self.args.norig else not_aug_inputs
            if self.buffer.num_seen_examples < self.args.buffer_size:
                self.buffer.add_data(examples=inputs[:real_batch_size],
                                     labels=labels[:real_batch_size])
            else:
                inc_weight = real_batch_size / self.buffer.num_seen_examples
                buf_inputs, buf_labels = self.buffer.get_all_data()
                inputs = torch.cat((inputs[:real_batch_size], buf_inputs))
                labels = torch.cat((labels[:real_batch_size], buf_labels))
                chosen_indexes = self.ifs.select(inputs.cpu(), labels.cpu(), self.buffer.buffer_size, self.kernel_fn,
                                                 self.args.lmbda, self.args.mu, self.args.nu, inc_weight)[0]
                out_indexes = np.setdiff1d(np.arange(self.buffer.buffer_size), chosen_indexes - real_batch_size)
                in_indexes = chosen_indexes[chosen_indexes < real_batch_size]
                self.buffer.replace_data(out_indexes, inputs[in_indexes], labels[in_indexes])
                self.buffer.num_seen_examples += real_batch_size

        return loss.item()
