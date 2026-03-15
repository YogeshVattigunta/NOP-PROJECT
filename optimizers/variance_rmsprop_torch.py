import torch
from torch.optim import Optimizer

class VarianceRMSProp(Optimizer):
    def __init__(self, params, lr=1e-4, beta=0.9, alpha=0.03, epsilon=1e-8, v_min=1e-10, v_max=10.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
            
        defaults = dict(lr=lr, beta=beta, alpha=alpha, epsilon=epsilon, v_min=v_min, v_max=v_max)
        super(VarianceRMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p.data)

                v = state['v']
                beta = group['beta']
                state['step'] += 1

                # Bounded EMA of squared gradients
                # Avoid Adagrad-style unlimited growth by clamping
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                v.clamp_(min=group['v_min'], max=group['v_max'])

                # Variance tracking term
                # sigma_sq = v_t - (beta * v_{t-1})^2
                # Note: This is an approximation of the variance tracker
                sigma_sq = v - (beta * v)**2
                sigma_sq.clamp_(min=group['epsilon'])

                # Preconditioned update
                # theta = theta - (lr / sqrt(sigma_sq + eps)) * grad
                p.data.addcdiv_(grad, torch.sqrt(sigma_sq).add_(group['epsilon']), value=-group['lr'])

        return loss
