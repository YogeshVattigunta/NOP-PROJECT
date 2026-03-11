import torch
from torch.optim.optimizer import Optimizer, required

class VarianceRMSProp(Optimizer):
    """
    Custom VarianceRMSProp optimizer.
    Incorporates gradient variance tracking to modify the accumulator to stabilize
    gradient variance and improve learning for imbalanced datasets.
    """
    def __init__(self, params, lr=1e-2, alpha=0.1, beta=0.9, eps=1e-8, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps, weight_decay=weight_decay)
        super(VarianceRMSProp, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("VarianceRMSProp does not support sparse gradients")
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grad_mean'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                grad_mean = state['grad_mean']

                state['step'] += 1

                # Update mean gradient: mu_t = beta * mu_(t-1) + (1-beta) * g_t
                grad_mean.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute variance: sigma_t = (g_t - mu_t)^2
                variance = (grad - grad_mean).pow(2)

                # Update the accumulator: v_t = beta * v_(t-1) + (1-beta) * (g_t^2 + alpha * sigma_t)
                square_avg.mul_(beta).addcmul_(grad, grad, value=1 - beta).add_(variance, alpha=(1 - beta) * alpha)

                # Weight update: w = w - lr * g_t / sqrt(v_t + eps)
                avg = square_avg.sqrt().add_(eps)
                p.addcdiv_(grad, avg, value=-lr)

        return loss
