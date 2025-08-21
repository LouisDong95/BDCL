import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            return torch.tensor([0.0]).cuda()
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def mask_correlated_samples(self, N):
        mask = torch.ones((N//2, N//2))
        mask = mask.fill_diagonal_(0)
        # for i in range(N//2):
        #     mask[i, N//2 + i] = 0
        #     mask[N//2 + i, i] = 0
        mask = torch.cat((mask, mask), 1)
        mask = torch.cat((mask, mask), 0)
        mask = mask.bool()
        return mask

    # Instance Contrastive Loss
    def forward_instance(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        q_i = self.softmax(q_i)
        q_j = self.softmax(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    # Feature Constraint
    def forward_feature(self, z_i):
        b, d = z_i.size()
        z_i = F.normalize(z_i, dim=0)
        sim = torch.matmul(z_i.t(), z_i)
        loss = self.mse(sim, torch.eye(d).to('cuda'))
        return loss

    # Cluster Constraint
    def forward_cluster(self, q):
        q_prob = F.softmax(q, 1)
        b, k = q_prob.size()
        q_prob = F.normalize(q_prob, dim=0)
        sim = torch.matmul(q_prob.t(), q_prob)
        loss = self.mse(sim, torch.eye(k).to('cuda'))
        return loss

    # Consistency Loss
    def forward_consistency(self, qi, qi_n, qj, qj_n):
        qi_prob = self.softmax(qi)
        qi_n_prob = self.softmax(qi_n)
        qj_prob = self.softmax(qj)
        qj_n_prob = self.softmax(qj_n)
        #
        # p_i = qi_prob.sum(0).view(-1)
        # p_i /= p_i.sum()
        # ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        # p_j = qj_prob.sum(0).view(-1)
        # p_j /= p_j.sum()
        # ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        # entropy = ne_i + ne_j

        # sim1 = torch.bmm(qj_n_prob.unsqueeze(1), qi_prob.unsqueeze(2)).squeeze()
        sim1 = torch.bmm(qi_prob.unsqueeze(1), qj_prob.unsqueeze(2)).squeeze()
        sim2 = torch.bmm(qi_n_prob.unsqueeze(1), qj_prob.unsqueeze(2)).squeeze()
        ones = torch.ones_like(sim1)
        consistency_loss = (self.bce(sim1, ones) + self.bce(sim2, ones))/2
        return consistency_loss

    def entropy(self, q):
        q_prob = self.softmax(q)
        p = q_prob.sum(0).view(-1)
        p /= p.sum()
        loss = math.log(p.size(0)) + (p * torch.log(p)).sum()
        return loss * 2.0

    def self_label(self, q, p):
        q_prob = self.softmax(q)
        max_prob, target = torch.max(q_prob, dim=1)
        mask = max_prob > 0.99
        b, c = q_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        idx, counts = torch.unique(target_masked, return_counts=True)
        freq = 1 / (counts.float() / n)
        weight = torch.ones(c).cuda()
        weight[idx] = freq

        loss = self.loss(p, target, mask, weight=weight, reduction='mean')
        return loss


