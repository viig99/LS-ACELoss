import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ACELabelSmoothingLoss(nn.Module):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets, input_lengths, target_lengths):
        T_, bs, class_size = logits.size()
        tagets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(tagets_split, batch_first=True, padding_value=0)
        targets_padded = F.one_hot(targets_padded.long(), num_classes=class_size) # batch, seq, class
        targets_padded = (targets_padded * (1-self.alpha)) + (self.alpha/class_size)
        targets_padded = torch.sum(targets_padded, 1).float().cuda() # sum across seq, to get batch * class    
        targets_padded[:,0] = T_ - target_lengths
        probs = torch.softmax(logits, dim=2) # softmax on class
        probs = torch.sum(probs, 0) # sum across seq, to get batch * class
        probs = probs/T_
        targets_padded = targets_padded/T_
        targets_padded = F.normalize(targets_padded, p=1, dim=1)
        return F.kl_div(torch.log(probs), targets_padded, reduction='batchmean')

class SentenceCrossEntropy(nn.Module):

    def __init__(self, label_smoothing=0.1):
        super(SentenceCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.PAD_INDEX = 0

    def forward(self, logits, targets, input_lengths, target_lengths):
        T_, bs, class_size = logits.size()
        targets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets_split, batch_first=True, padding_value=self.PAD_INDEX)
        targets_padded = F.pad(input=targets_padded, pad=(0, T_ - targets_padded.shape[1]), mode='constant', value=self.PAD_INDEX)
        mask = targets_padded != self.PAD_INDEX
        targets_padded = F.one_hot(targets_padded.long(), num_classes=class_size).to(logits.device) # batch, seq, class
        targets_padded = targets_padded.mul(1.0 - self.label_smoothing) + (1 - targets_padded).mul(self.label_smoothing / (class_size - 1))
        loss = -(F.log_softmax(logits.permute(1, 2, 0), dim=1).mul(targets_padded.permute(0, 2, 1)).sum(1))
        loss = loss.masked_select(mask.to(loss.device)).mean()
        return loss

class FocalSentenceCrossEntropy(SentenceCrossEntropy):

    def __init__(self, alpha=0.25, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = super().forward(log_probs, targets, input_lengths, target_lengths)
        p = torch.exp(-loss)
        return self.alpha * torch.pow((1-p), self.gamma) * loss

class FocalCTCLoss(nn.CTCLoss):
    def __init__(self, reduction='mean', zero_infinity=True, alpha=0.25, gamma=0.5):
        super(FocalCTCLoss, self).__init__(
            reduction=reduction, zero_infinity=zero_infinity)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, input_lengths, target_lengths):
        ctc_loss_param = F.ctc_loss(torch.log_softmax(logits, dim=2), targets, input_lengths,
                                    target_lengths, self.blank, self.reduction, self.zero_infinity)
        p = torch.exp(-ctc_loss_param)
        return self.alpha * torch.pow((1-p), self.gamma) * ctc_loss_param