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