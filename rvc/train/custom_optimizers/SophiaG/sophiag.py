import torch
from torch.optim import Optimizer
import math

class SophiaG(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.8, 0.99), rho=0.04, weight_decay=0, eps=1e-8, *, maximize: bool = False):
        """
        Initialize Sophia-G optimizer with decoupled weight decay.
        Args:
            params: model parameters
            lr: learning rate
            betas: coefficients used for computing running averages of gradient and its square
            rho: parameter for updating history
            weight_decay: weight decay (L2 penalty, applied decoupled)
            eps: small value for numerical stability
            maximize: maximize the objective (default: False)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps, maximize=maximize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            rho = group['rho']
            weight_decay = group['weight_decay']
            eps = group['eps']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad if not maximize else -p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, hessian = state['exp_avg'], state['hessian']
                step = state['step']
                state['step'] += 1

                # Gradient scaling using clamp
                grad_norm = torch.linalg.vector_norm(grad)
                grad = grad.clamp(min=-1.0, max=1.0) / grad_norm.clamp(min=1.0)

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if step % 2 == 1:
                    # Precompute beta2_pow and beta1_pow for the step
                    beta2_pow = beta2 ** step
                    beta1_pow = beta1 ** step

                    # Update Hessian diagonal approximation
                    hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute step_size / bias_correction
                    step_size = lr * math.sqrt(1 - beta2_pow) / (1 - beta1_pow)

                    # Compute denominator with hessian, rho, and eps
                    denom = hessian.sqrt().add_(rho).add_(eps)

                    # Update parameters
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # Apply decoupled weight decay
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)

        return loss