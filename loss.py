import numpy as np
import torch
import sys


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)
    # eye = torch.eye(anchor.size(0)).cuda()
    eps = 1e-6
    # the clamp function is added to prevent sqrt(0) in loss_myloss_second
    return torch.sqrt(torch.clamp(d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                       - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0), min=0.0) + eps)


def distance_vectors_pairwise(anchor, positive, negative=None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2 * torch.sum(anchor * positive, dim=1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2 * torch.sum(anchor * negative, dim=1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2 * torch.sum(positive * negative, dim=1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p


def loss_myloss(anchor, positive, neighbor=8, margin=1.0, anchor_swap=True, square=True):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.eye(dist_matrix.size(1)).cuda()
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10

    # torch.ge比较大小，greater or equal，返回bool量，这里是比较整个矩阵中的值的大小，距离大的项反应在mask中为0，距离小的为1
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    min_horizontal = torch.topk(dist_without_min_on_diag, k=neighbor, dim=1, largest=False)[0]
    horizontal_threshold = min_horizontal[:, -1].view(-1, 1).repeat((1, min_horizontal.shape[1]))
    horizontal_weight = horizontal_threshold / min_horizontal
    if square:
        horizontal_weight = horizontal_weight * horizontal_weight
    horizontal_weight = horizontal_weight / horizontal_weight.sum(dim=1, keepdim=True)
    min_horizontal_value = min_horizontal * horizontal_weight
    min_neg = torch.sum(min_horizontal_value, dim=1)

    if anchor_swap:
        min_vetical = torch.topk(dist_without_min_on_diag, k=neighbor, dim=0, largest=False)[0]
        vetical_threshold = min_vetical[-1, :].view(1, -1).repeat((min_vetical.shape[0], 1))
        vetical_weight = vetical_threshold / min_vetical
        if square:
            vetical_weight = vetical_weight * vetical_weight
        vetical_weight = vetical_weight / vetical_weight.sum(dim=0, keepdim=True)
        min_vetical_value = min_vetical * vetical_weight
        min_neg2 = torch.sum(min_vetical_value, dim=0)
        min_neg = torch.min(min_neg, min_neg2)

    loss = torch.clamp(margin + pos - min_neg, min=0.0)
    loss = torch.mean(loss)
    return loss


def loss_myloss_second(anchor, positive, neighbor=8, square=True, anchor_swap=True):
    """
    这里基于SOSNet做一个调整，在hardnet的loss函数中，计算了anchor与positive的元素间相互距离，我们这里计算anchor与positive内部的相互距离，
    得到两个距离矩阵，anchor_mat与positive_mat，其中anchor_mat[i, j] = distance(anchor[i], anchor[j]), positive_mat有着类似的关系。
    positive_mat[i,j] = distance(positive[i], positive[j])。在loss_hardnet中，由于有的class中可能有大于两个的点对应，具体如如下代码
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    因此需要做干扰排除，也就是在同时不用顾及到有另外的干扰项，因此不用做干扰排除。
    同时，考虑到SOSNet中所有距离都采用同样的权重，我们认为更近的点应该有更大的权重作为区分。因此我们将n个neighbor进行排序，
    设anchor与positive的距离分别为：a1,a2,...,an;    p1,p2,...,pn; 做差之后再排序，从小到大为d1,d2,...,dn;
    那么loss = ∑(an/ai)|ai-pi|
    :param anchor:
    :param positive:
    :param anchor_swap:
    :param margin:
    :param neighbor:
    :return:
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    anchor_mat = distance_matrix_vector(anchor, anchor) + eps
    positive_mat = distance_matrix_vector(positive, positive) + eps
    anchor_eye = torch.eye(anchor_mat.size(1)).cuda()
    positive_eye = torch.eye(positive_mat.size(1)).cuda()

    anchor_without_min_on_diag = anchor_mat + anchor_eye * 10
    positive_without_min_on_diag = positive_mat + positive_eye * 10

    # 找到前n项
    anchor_value, anchor_index = torch.topk(anchor_without_min_on_diag, k=neighbor, dim=1, largest=False)
    positive_dist = torch.zeros(anchor_value.shape).cuda()
    for i in range(anchor.shape[0]):
        positive_dist[i] = positive_without_min_on_diag[i, anchor_index[i, :]]

    anchor_threshold = anchor_value[:, -1].view(-1, 1).repeat((1, anchor_value.shape[1]))
    anchor_weight = anchor_threshold / anchor_value
    if square:
        anchor_weight = anchor_weight * anchor_weight
    anchor_weight = anchor_weight / anchor_weight.sum(dim=1, keepdim=True)

    min_neg1 = anchor_weight * torch.abs(anchor_value - positive_dist)
    min_neg1 = min_neg1.sum(dim=1)
    loss = torch.mean(min_neg1)

    if anchor_swap:
        positive_value, positive_index = torch.topk(positive_without_min_on_diag, k=neighbor, dim=1, largest=False)
        anchor_dist = torch.zeros(positive_value.shape).cuda()
        for i in range(positive.shape[0]):
            anchor_dist[i] = anchor_without_min_on_diag[i, positive_index[i, :]]
        positive_threshold = positive_value[:, -1].view(-1, 1).repeat((1, positive_value.shape[1]))
        positive_weight = positive_threshold / positive_value
        if square:
            positive_weight = positive_weight * positive_weight
        positive_weight = positive_weight / positive_weight.sum(dim=1, keepdim=True)
        min_neg2 = positive_weight * torch.abs(positive_value - anchor_dist)
        min_neg2 = min_neg2.sum(dim=1)
        loss += torch.mean(min_neg2)
    return loss


def loss_adaption(anchor, positive, mag=4, margin=1.0, anchor_swap=True, square=True, clamp=8):
    """
    自适应采样loss函数，在这里，我们设置了mag作为倍数衡量，设anchor与positive的距离分别为a1,a2,...,an(距离从小到大排序)，
    相比loss_myloss，我们不使用knn考虑固定的邻居数量，而是考虑距离在一定范围内的邻居数量，以a1为基准，有一个阈值threshold=mag * a1
    我们只考虑距离小于这一阈值范围内的邻居。
    :param anchor:
    :param positive:
    :param mag: 倍数阈值，我们只考虑距离与最小距离的比值小于这一倍数的邻居
    :param anchor_swap:
    :return:
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    # torch.sort()
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.eye(dist_matrix.size(1)).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10

    # torch.ge比较大小，greater or equal，返回bool量，这里是比较整个矩阵中的值的大小，距离大的项反应在mask中为0，距离小的为1
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    horizontal_dist = torch.sort(dist_without_min_on_diag, dim=1)[0]
    horizontal_threshold = (horizontal_dist[:, 0] * mag).view(-1, 1).repeat((1, horizontal_dist.shape[1]))
    horizontal_weight = horizontal_threshold / horizontal_dist
    horizontal_weight[horizontal_weight < 1 - eps] = 0
    horizontal_weight[:, clamp:] = 0
    if square:
        horizontal_weight = horizontal_weight * horizontal_weight
    horizontal_weight = horizontal_weight / horizontal_weight.sum(dim=1, keepdim=True)
    min_neg = horizontal_weight * horizontal_dist
    min_neg = min_neg.sum(dim=1)

    if anchor_swap:
        vetical_dist = torch.sort(dist_without_min_on_diag, dim=0)[0]
        vetical_threshold = (vetical_dist[0, :] * mag).view(1, -1).repeat((dist_matrix.shape[0], 1))
        vetical_weight = vetical_threshold / vetical_dist
        vetical_weight[vetical_weight < 1 - eps] = 0
        vetical_weight[clamp:, :] = 0
        if square:
            vetical_weight = vetical_weight * vetical_weight
        vetical_weight = vetical_weight / vetical_weight.sum(dim=0, keepdim=True)
        min_neg2 = vetical_weight * vetical_dist
        min_neg2 = min_neg2.sum(dim=0)
        min_neg = torch.min(min_neg, min_neg2)

    loss = torch.clamp(margin + pos - min_neg, min=0.0)
    loss = torch.mean(loss)
    return loss


def loss_adaption_second(anchor, positive, mag=4, square=True, anchor_swap=True, clamp=8):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    anchor_mat = distance_matrix_vector(anchor, anchor) + eps
    positive_mat = distance_matrix_vector(positive, positive) + eps
    anchor_eye = torch.eye(anchor_mat.size(1)).cuda()
    positive_eye = torch.eye(positive_mat.size(1)).cuda()

    anchor_without_min_on_diag = anchor_mat + anchor_eye * 10
    positive_without_min_on_diag = positive_mat + positive_eye * 10

    horizontal_dist, horizontal_index = torch.sort(anchor_without_min_on_diag, dim=1)
    positive_value = torch.zeros(horizontal_dist.shape).cuda()
    for i in range(positive_value.shape[0]):
        positive_value[i] = positive_without_min_on_diag[i][horizontal_index[i]]
    horizontal_threshold = (horizontal_dist[:, 0] * mag).view(-1, 1).repeat((1, horizontal_dist.shape[1]))
    horizontal_weight = horizontal_threshold / horizontal_dist
    horizontal_weight[horizontal_weight < 1 - eps] = 0
    horizontal_weight[:, clamp:] = 0
    if square:
        horizontal_weight = horizontal_weight * horizontal_weight
    horizontal_weight = horizontal_weight / horizontal_weight.sum(dim=1, keepdim=True)

    min_neg1 = horizontal_weight * torch.abs(horizontal_dist - positive_value)
    min_neg1 = min_neg1.sum(dim=1)
    loss = torch.mean(min_neg1)

    if anchor_swap:
        vetical_dist, vetical_index = torch.sort(positive_without_min_on_diag, dim=1)
        anchor_value = torch.zeros(vetical_dist.shape).cuda()
        for i in range(anchor_value.shape[0]):
            anchor_value[i] = anchor_without_min_on_diag[i][vetical_index[i]]
        vetical_threshold = (vetical_dist[:, 0] * mag).view(-1, 1).repeat((1, vetical_dist.shape[1]))
        vetical_weight = vetical_threshold / vetical_dist
        vetical_weight[vetical_weight < 1 - eps] = 0
        vetical_weight[:, clamp:] = 0
        if square:
            vetical_weight = vetical_weight * vetical_weight
        vetical_weight = vetical_weight / vetical_weight.sum(dim=1, keepdim=True)
        min_neg2 = vetical_weight * torch.abs(vetical_dist - anchor_value)
        min_neg2 = min_neg2.sum(dim=1)
        loss += torch.mean(min_neg2)
    return loss


def loss_random_sampling(anchor, positive, negative, anchor_swap=False, margin=1.0, loss_type="triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
        min_neg = torch.min(d_a_n, d_p_n)
    else:
        min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = - torch.log(exp_pos / exp_den)
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else:
        print('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


def loss_L2Net(anchor, positive, anchor_swap=True, margin=1.0, loss_type="triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.eye(dist_matrix.size(1)).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10
    mask = (dist_without_min_on_diag.ge(0.008) - 1) * -1
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    if loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos1);
        exp_den = torch.sum(torch.exp(2.0 - dist_matrix), 1) + eps
        loss = -torch.log(exp_pos / exp_den)
        if anchor_swap:
            exp_den1 = torch.sum(torch.exp(2.0 - dist_matrix), 0) + eps
            loss += -torch.log(exp_pos / exp_den1)
    else:
        print('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


def loss_HardNet(anchor, positive, anchor_swap=True, margin=1.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.eye(dist_matrix.size(1)).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10
    # torch.ge比较大小，greater or equal，返回bool量，这里是比较整个矩阵中的值的大小，距离大的项反应在mask中为0，距离小的为1
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    min_neg = torch.min(dist_without_min_on_diag, 1)[0]
    if anchor_swap:
        min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
        min_neg = torch.min(min_neg, min_neg2)

    min_neg = min_neg
    pos = pos1
    loss = torch.clamp(margin + pos - min_neg, min=0.0)
    loss = torch.mean(loss)
    return loss


def global_orthogonal_regularization(anchor, negative):
    neg_dis = torch.sum(torch.mul(anchor, negative), 1)
    dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis), 2) + torch.clamp(torch.mean(torch.pow(neg_dis, 2)) - 1.0 / dim, min=0.0)

    return gor


def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)
