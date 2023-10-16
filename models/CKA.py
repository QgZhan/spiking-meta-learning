# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: CKA.py
@time: 2021/2/3 17:27
"""

import torch


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
    x: A num_examples x num_features matrix of features.

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    return torch.bmm(x, x.permute(0, 2, 1))


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = torch.bmm(x, x.permute(0, 2, 1))  # (10, 64, 64)  # torch.mm(x, torch.t(x))
    sq_norms = torch.diagonal(dot_products, dim1=-2, dim2=-1)  # (10, 64)  # torch.diag(dot_products)
    sq_distances = (sq_norms[:, :, None] - dot_products) + (sq_norms[:, None, :] - dot_products)  # (10, 1, 1)
    if not float(torch.max(sq_distances.permute(0, 2, 1) - sq_distances)) < 1e-6:
        raise ValueError('sq_distances must be a symmetric matrix.')
    sq_median_distance = torch.median(sq_distances.view(sq_distances.shape[0], -1), dim=-1, keepdim=True)[0]  # (10, 1)
    # sq_median_distance = sq_median_distance.unsqueeze(-1)
    result = torch.exp(torch.div(-sq_distances, (2 * threshold ** 2 * sq_median_distance.unsqueeze(-1) + 1e-7)))  # (10, 1, 1)
    return result


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
    gram: A num_examples x num_examples symmetric matrix.  对称矩阵
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

    Returns:
    A symmetric matrix with centered columns and rows.
    """

    if not float(torch.max(gram.permute(0, 2, 1) - gram)) < 1e-6:
        raise ValueError('Input must be a symmetric matrix.')
    # if not float(torch.max(torch.t(gram) - gram)) < 1e-6:
    #     raise ValueError('Input must be a symmetric matrix.')
    gram = gram.clone()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        gram = gram - torch.diag(torch.diag(gram))
        means = torch.sum(gram, 0).float() / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram = gram - torch.diag(torch.diag(gram))
    else:
        means = torch.mean(gram, 1)
        means -= torch.mean(means, 1, keepdim=True) / 2
        gram -= means[:, :, None]
        gram -= means[:, None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):  # 要求输入是对称矩阵
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)  # (query_num, 1, 1)
    gram_y = center_gram(gram_y, unbiased=debiased)  # (support_num, 1, 1)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = torch.matmul(gram_x.view(gram_x.shape[0], 1, -1).unsqueeze(1).repeat(1, gram_y.shape[0], 1, 1),
                               gram_y.view(gram_y.shape[0], 1, -1).permute(0, 2, 1)).squeeze(-1)  # (query_num, support_num, 1)

    normalization_x = torch.norm(torch.norm(gram_x, dim=-1), dim=-1, keepdim=True)  # (query_num, 1)
    normalization_y = torch.norm(torch.norm(gram_y, dim=-1), dim=-1, keepdim=True)  # (support_num, 1)
    normalization_xy = normalization_x.unsqueeze(1).repeat(1, normalization_y.shape[0], 1) * normalization_y

    return torch.div(scaled_hsic, normalization_xy).squeeze(-1)
    # return scaled_hsic / (normalization_x * normalization_y)


def linear_CKA(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_linear(gram_x), unbiased=debiased)
    gram_y = center_gram(gram_linear(gram_y),  unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    # scaled_hsic = torch.mm(gram_x.view(1, -1), torch.t(gram_y.view(1, -1)))
    #
    # normalization_x = torch.norm(gram_x)
    # normalization_y = torch.norm(gram_y)
    # return scaled_hsic / (normalization_x * normalization_y + 1e-7)

    scaled_hsic = torch.matmul(gram_x.view(gram_x.shape[0], 1, -1).unsqueeze(1).repeat(1, gram_y.shape[0], 1, 1),
                               gram_y.view(gram_y.shape[0], 1, -1).permute(0, 2, 1)).squeeze(-1)  # (query_num, support_num, 1)

    normalization_x = torch.norm(torch.norm(gram_x, dim=-1), dim=-1, keepdim=True)  # (query_num, 1)
    normalization_y = torch.norm(torch.norm(gram_y, dim=-1), dim=-1, keepdim=True)  # (support_num, 1)
    normalization_xy = normalization_x.unsqueeze(1).repeat(1, normalization_y.shape[0], 1) * normalization_y + 1e-7

    return torch.div(scaled_hsic, normalization_xy).squeeze(-1)


def rbf_CKA(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_rbf(gram_x), unbiased=debiased)
    gram_y = center_gram(gram_rbf(gram_y), unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    # scaled_hsic = torch.mm(gram_x.view(1, -1), torch.t(gram_y.view(1, -1)))
    #
    # normalization_x = torch.norm(gram_x)
    # normalization_y = torch.norm(gram_y)
    # return scaled_hsic / (normalization_x * normalization_y)

    scaled_hsic = torch.matmul(gram_x.view(gram_x.shape[0], 1, -1).unsqueeze(1).repeat(1, gram_y.shape[0], 1, 1),
                               gram_y.view(gram_y.shape[0], 1, -1).permute(0, 2, 1)).squeeze(-1)  # (query_num, support_num, 1)

    normalization_x = torch.norm(torch.norm(gram_x, dim=-1), dim=-1, keepdim=True)  # (query_num, 1)
    normalization_y = torch.norm(torch.norm(gram_y, dim=-1), dim=-1, keepdim=True)  # (support_num, 1)
    normalization_xy = normalization_x.unsqueeze(1).repeat(1, normalization_y.shape[0], 1) * normalization_y + 1e-7

    return torch.div(scaled_hsic, normalization_xy).squeeze(-1)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * torch.sum(sum_squared_rows_x * sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

    Returns:
    The value of CKA between X and Y.
    """
    features_x = features_x - torch.mean(features_x, 0, True)
    features_y = features_y - torch.mean(features_y, 0, True)

    dot_product_similarity = torch.norm(torch.mm(torch.t(features_x), features_y)) ** 2
    normalization_x = torch.norm(torch.mm(torch.t(features_x), features_x))
    normalization_y = torch.norm(torch.mm(torch.t(features_y), features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = torch.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = torch.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = torch.sum(sum_squared_rows_x)
        squared_norm_y = torch.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = torch.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)