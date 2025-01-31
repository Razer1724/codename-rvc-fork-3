import math
import torch
from torch.optim.optimizer import Optimizer

class RAdam(Optimizer):
    """
    RAdam Optimizer
    
    Implementation of RAdam optimizer from paper:
    'On the Variance of the Adaptive Learning Rate and Beyond'
    https://arxiv.org/abs/1908.03265
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        degenerated_to_sgd: whether to degenerate back to SGD when variance is too high (default: True)
    """

    def __init__(self, params, lr=1e-4, betas=(0.8, 0.99), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       degenerated_to_sgd=degenerated_to_sgd)
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
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
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute length of the approximate SMA
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # Variance rectification term
                if N_sma >= 5:
                    # Compute the variance rectification term
                    rect_term = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * 
                                        (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2))
                    # Apply momentum with variance rectification
                    step_size = group['lr'] * rect_term
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                elif group['degenerated_to_sgd']:
                    # Degenerate to SGD
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p.add_(exp_avg, alpha=-step_size)

        return loss