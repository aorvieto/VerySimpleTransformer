import torch
import numpy as np

def flat_params(m):
    flat_data = []
    for p in m.parameters():
        flat_data.append(p.data.view(-1))
    return torch.cat(flat_data)

def get_grad_norm_squared(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm

def zero_gradients(x):
    if x.grad is not None:
    	x.grad.zero_()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cosine_wa_lr(target, global_step, total_steps, warmup_steps):
	cosine_lr = 0.5 * target * (1 + np.cos(np.pi * (global_step - warmup_steps) / float(total_steps - warmup_steps)))
	if warmup_steps == 0:
		return cosine_lr.item()
	else:
		warmup_lr = np.array(target * (global_step / warmup_steps))
		learning_rate = np.where(global_step < warmup_steps, warmup_lr, cosine_lr)
		return learning_rate.item()