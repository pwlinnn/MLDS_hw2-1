import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = 0
        self.avg_loss = 0
    def forward(self, prediction, groundtruth, lengths):
        """
        @param lengths: valid length of each caption 
        """
        batch_size = prediction.shape[0]
        pred_cat = None
        gt_cat = None
        for i in range(batch_size):
            pred = prediction[i]
            gt = groundtruth[i]
            tgt_len = lengths[i]-1 

            pred = pred[:tgt_len]
            gt = gt[:tgt_len]

            if i == 0:
                pred_cat = pred
                gt_cat = gt
            else:
                pred_cat = torch.cat((pred_cat, pred), dim=0)
                gt_cat = torch.cat((gt_cat, gt), dim=0)

        self.loss = self.loss_fn(pred_cat, gt_cat)
        self.avg_loss = self.loss/batch_size
        return self.loss
            
if __name__ == '__main__':
    from torch.autograd import Variable

    x = torch.rand(32, 16, 1799)
    torch.Tensor(x)
    y = torch.LongTensor(32, 16).random_(1799)
    length = list(torch.LongTensor(32).random_(5, 17))
    length.sort(reverse=True)
    x.requires_grad = True
    #x, y = Variable(x, requires_grad=True), Variable(y)

    x = x+2

    print(x)
    print(y)
    print(length)

    ll = CustomLoss()

    loss = ll(x, y, length)
    loss.backward()

    print(loss)