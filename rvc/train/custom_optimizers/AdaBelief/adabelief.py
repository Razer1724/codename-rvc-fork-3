import math
import torch
from torch.optim.optimizer import Optimizer

class AdaBelief(Optimizer):
    """
    AdaBelief Optimizer
    
    Implementation of AdaBelief optimizer from paper:
    'AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients'
    https://arxiv.org/abs/2010.07468
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-16)
        weight_decay: weight decay (L2 penalty) (default: 0)
        rectify: whether to enable the rectification term from RAdam (default: True)
        degenerated_to_sgd: whether to degenerate back to SGD when variance is too high (default: True)
    """

    def __init__(self, params, lr=1e-4, betas=(0.8, 0.99), eps=1e-8,
                 weight_decay=0, rectify=True, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       rectify=rectify, degenerated_to_sgd=degenerated_to_sgd)
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('rectify', True)
            group.setdefault('degenerated_to_sgd', True)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                # Rectification term from RAdam
                if group['rectify']:
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])
                    
                    if rho_t > 4:  # Conservative PPL setting from RAdam
                        r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        step_size = group['lr'] * r_t * math.sqrt(bias_correction2) / bias_correction1
                    elif group['degenerated_to_sgd']:
                        step_size = group['lr'] / bias_correction1
                    else:
                        step_size = group['lr']
                else:
                    step_size = group['lr']

                # Update parameters
                denom = (exp_avg_var.add_(group['eps'])).sqrt_()
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss