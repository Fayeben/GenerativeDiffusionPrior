import ignite
import torch
import torch.nn as nn

class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def compute_psnr_and_ssim(pred, target, device, crop_border=4):
    # pred, target and are images of shape HW3, uint8, range from 0 to 255
    pred = torch.from_numpy(pred).to(device).float()
    if crop_border != 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border, ...]
    pred = pred.permute(2,0,1).unsqueeze(0) #13HW

    target = torch.from_numpy(target).to(device).float()
    if crop_border != 0:
        target = target[crop_border:-crop_border, crop_border:-crop_border, ...]
    target = target.permute(2,0,1).unsqueeze(0) #13HW

    model = DummyModule()
    default_evaluator =  ignite.engine.create_supervised_evaluator(model)

    psnr = ignite.metrics.PSNR(data_range=255, device=device)
    psnr.attach(default_evaluator, 'psnr')

    metric = ignite.metrics.SSIM(data_range=255, device=device)
    metric.attach(default_evaluator, 'ssim')

    state = default_evaluator.run([[pred, target]])

    psnr = state.metrics['psnr']
    ssim = state.metrics['ssim']
    return psnr, ssim

import pdb
if __name__ == '__main__':

    device= torch.device('cuda')
    model = DummyModule()
    default_evaluator =  ignite.engine.create_supervised_evaluator(model)

    psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    psnr.attach(default_evaluator, 'psnr')

    metric = ignite.metrics.SSIM(data_range=1.0, device=device)
    metric.attach(default_evaluator, 'ssim')

    preds = torch.rand([4, 3, 16, 16], device=device)
    target = preds * 0.75
    state = default_evaluator.run([[preds, target]])
    pdb.set_trace()
    