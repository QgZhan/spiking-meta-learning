import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from models import MyCKA, MMD, CKA
from torch.autograd import Variable
from qpth.qp import QPFunction

from . import few_shot


_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        # if remove and (basename.startswith('_')
        #         or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, proto_label=None, n_way=None, n_shot=None, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':      # .t()转置
            logits = torch.mm(feat, proto.t())   # 矩阵相乘，a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵

        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(dim=-1)

        elif metric == 'CKA':     ######################################################
            feat = feat.squeeze().cuda()     # [75, 512]
            proto = proto.squeeze().cuda()    # [5, 512]
            logits = []
            for i in range(0, feat.shape[0]):
                for j in range(0, proto.shape[0]):
                    # logit = MyCKA.linear_CKA(feat[i].unsqueeze(1), proto[j].unsqueeze(1))
                    logit = MyCKA.RBF_CKA(feat[i].unsqueeze(1), proto[j].unsqueeze(1))
                    logits.append(logit.cuda())
            logits = torch.tensor(logits).cuda()
            logits = logits.reshape(feat.shape[0], proto.shape[0])

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))       # torch.Size([1, 75, 5])

        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) - proto.unsqueeze(1)).pow(2).sum(dim=-1)

        elif metric == 'SVM':
            logits = MetaOptNetHead_SVM_CS(feat, proto, proto_label, n_way, n_shot)

    elif feat.dim() == 4:
        if metric == 'CKA':
            task_num = feat.shape[0]
            task_logits = []
            for task_id in range(task_num):
                logit = CKA.rbf_CKA(feat[task_id], proto[task_id])
                task_logits.append(logit)
            logits = torch.stack(task_logits).cuda()

    return logits * temp


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert (A.dim() == 3)
    assert (B.dim() == 3)
    assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1, 2))


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and
    returns the classification score on the query set.

    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).

    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    # query: (1, 75, 512)
    # support: (1, 25, 512)
    # n_way = 5
    # n_shot = 5

    tasks_per_batch = query.size(0)  # 1
    n_support = support.size(1)  # 25
    n_query = query.size(1)  # 75

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    # Here we solve the dual problem:
    # Note that the classes are indexed by m & samples are indexed by i.
    # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    # s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    # where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    # and C^m_i = C if m  = y_i,
    # C^m_i = 0 if m != y_i.
    # This borrows the notation of liblinear.

    # \alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)  # (1, 25, 25)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()  # (1, 5, 5)
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)  # (1, 125, 125)
    # This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support,
                                                                     n_way * n_support).cuda()    # (1, 125, 125)

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support),
                                     n_way)  # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)  # (1, 125)

    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    # print (G.size())
    # This part is for the inequality constraints:
    # \alpha^m_i <= C^m_i \forall m,i
    # where C^m_i = C if m  = y_i,
    # C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)  # (1, 125, 125)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    # print (C.size(), h.size())
    # This part is for the equality constraints:
    # \sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()  # (1, 25, 25)

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    # print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())  # (1, 125)

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)  # (1, 25, 75)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)  # (1, 25, 75, 5)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)  # (1, 25, 5)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility  # (1, 25, 75, 5)
    logits = torch.sum(logits, 1)  # (1, 75, 5)

    return logits   # (1, 75, 5)


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=-1) == label).float()   # 返回一个维度上张量最大值的索引
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

