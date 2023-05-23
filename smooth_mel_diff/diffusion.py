import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import copy

class DiffusionSampler():

    # from https://github.com/Rongjiehuang/FastDiff
    noise_schedules = {
        "original": torch.linspace(
            0.000001,
            0.01,
            1000
        ),
        1000: torch.linspace(0.000001, 0.01, 1000),
        200: torch.linspace(0.0001, 0.02, 200),
        100: torch.linspace(0.0001, 0.04, 100),
        8: [
            6.689325005027058e-07,
            1.0033881153503899e-05,
            0.00015496854030061513,
            0.002387222135439515,
            0.035597629845142365,
            0.3681158423423767,
            0.4735414385795593,
            0.5,
        ],
        6: [
            1.7838445955931093e-06,
            2.7984189728158526e-05,
            0.00043231004383414984,
            0.006634317338466644,
            0.09357017278671265,
            0.6000000238418579
        ],
        4: [
            3.2176e-04,
            2.5743e-03,
            2.5376e-02,
            7.0414e-01
        ],
        3: [
            9.0000e-05,
            9.0000e-03,
            6.0000e-01
        ]
    }

    def __init__(self, model, diffusion_params):
        self.model = model
        self.diff_params = diffusion_params

    def __call__(self, batch, N=4, return_sequence=False, verbose=True):
        if N not in self.noise_schedules:
            raise ValueError(f"Invalid noise schedule length {N}")

        noise_schedule = self.noise_schedules[N]

        if not isinstance(noise_schedule, torch.Tensor):
            noise_schedule = torch.FloatTensor(noise_schedule)

        noise_schedule = noise_schedule.to(batch["cond_mel"].device)
        noise_schedule = noise_schedule.to(torch.float32)

        pred = self.sampling_given_noise_schedule(
            size=batch["cond_mel"].shape,
            inference_noise_schedule=noise_schedule,
            condition=batch["cond_mel"],
            return_sequence=return_sequence,
            verbose=verbose,
        )

        return pred

    def sampling_given_noise_schedule(
            self,
            size,
            inference_noise_schedule,
            condition,
            return_sequence=False,
            verbose=False,
        ):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the wavenet models
        size (tuple):                   size of tensor to be generated,
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors
        condition (torch.tensor):       ground truth mel spectrogram read from disk
                                        None if used for unconditional generation

        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        ddim = False

        _dh = self.diff_params
        T, alpha = _dh["T"], _dh["alpha"]
        assert len(alpha) == T
        assert len(size) == 3

        N = len(inference_noise_schedule)
        beta_infer = inference_noise_schedule
        alpha_infer = 1 - beta_infer
        sigma_infer = beta_infer + 0
        for n in range(1, N):
            alpha_infer[n] *= alpha_infer[n - 1]
            sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
        alpha_infer = torch.sqrt(alpha_infer)
        sigma_infer = torch.sqrt(sigma_infer)

        # Mapping noise scales to time steps
        steps_infer = []
        for n in range(N):
            step = DiffusionSampler.map_noise_scale_to_time_step(alpha_infer[n], alpha)
            if step >= 0:
                steps_infer.append(step)
        print(steps_infer, flush=True)
        steps_infer = torch.FloatTensor(steps_infer)

        # N may change since alpha_infer can be out of the range of alpha
        N = len(steps_infer)

        print('begin sampling, total number of reverse steps = %s' % N)

        x = torch.normal(0, 1, size=size).to(device=condition.device, dtype=condition.dtype)
        if return_sequence:
            x_ = copy.deepcopy(x)
            xs = [x_]
        with torch.no_grad():
            for n in range(N - 1, -1, -1):
                diffusion_steps = (steps_infer[n] * torch.ones((size[0], 1))).to(device=condition.device, dtype=condition.dtype)
                diffusion_steps = diffusion_steps.unsqueeze(1)
                epsilon_theta = self.model(cond_mel=condition, noisy_mel=x, step=diffusion_steps)
                if ddim:
                    alpha_next = alpha_infer[n] / (1 - beta_infer[n]).sqrt()
                    c1 = alpha_next / alpha_infer[n]
                    c2 = -(1 - alpha_infer[n] ** 2.).sqrt() * c1
                    c3 = (1 - alpha_next ** 2.).sqrt()
                    x = c1 * x + c2 * epsilon_theta + c3 * epsilon_theta  # std_normal(size)
                else:
                    x -= beta_infer[n] / torch.sqrt(1 - alpha_infer[n] ** 2.) * epsilon_theta
                    x /= torch.sqrt(1 - beta_infer[n])
                    if n > 0:
                        x = x + sigma_infer[n] * torch.normal(0, 1, size=size).to(device=condition.device, dtype=condition.dtype)
                if return_sequence:
                    x_ = copy.deepcopy(x)
                    xs.append(x_)
        if return_sequence:
            return xs
        return x

    @staticmethod
    def map_noise_scale_to_time_step(alpha_infer, alpha):
        if alpha_infer < alpha[-1]:
            return len(alpha) - 1
        if alpha_infer > alpha[0]:
            return 0
        for t in range(len(alpha) - 1):
            if alpha[t+1] <= alpha_infer <= alpha[t]:
                step_diff = alpha[t] - alpha_infer
                step_diff /= alpha[t] - alpha[t+1]
                return t + step_diff.item()
        return -1

def compute_diffusion_params(T, beta_0, beta_T):
    """
    Compute diffusion parameters from beta
    source: https://github.com/tencent-ailab/bddm/blob/2cebe0e6b7fd4ce8121a45d1194e2eb708597936/bddm/utils/diffusion_utils.py#L16
    """
    beta = torch.linspace(beta_0, beta_T, T)
    alpha = 1 - beta
    for t in range(1, len(beta)):
        alpha[t] *= alpha[t-1]
    alpha = torch.sqrt(alpha)
    diff_params = {
        "T": len(beta),
        "alpha": alpha,
    }
    return diff_params

def compute_diffusion_params_sigmoid(T, start, end, tau=1.0, clip_min=1e-9):
# A gamma function based on sigmoid function.
  alpha = torch.linspace(0, 1, T)
  v_start = torch.sigmoid(torch.tensor(start / tau))
  v_end = torch.sigmoid(torch.tensor(end / tau))
  output = torch.sigmoid((alpha * (end - start) + start) / tau)
  output = (v_end - output) / (v_end - v_start)
  return {
    "T": len(alpha),
    "alpha": torch.clip(output, clip_min, 1.),
  }

class StepEmbedding(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc_t1 = nn.Linear(
            dim_in,
            dim_hidden,
        )
        self.fc_t2 = nn.Linear(
            dim_hidden,
            dim_out,
        )
        self.dim_in = dim_in

        # init
        nn.init.xavier_uniform_(self.fc_t1.weight)
        nn.init.xavier_uniform_(self.fc_t2.weight)
        nn.init.zeros_(self.fc_t1.bias)
        nn.init.zeros_(self.fc_t2.bias)

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, steps):
        half_dim = self.dim_in // 2
        _embed = np.log(10000) / (half_dim - 1)
        _embed = torch.exp(torch.arange(half_dim) * -_embed)
        _embed = steps * _embed
        diff_embed = torch.cat(
            (torch.sin(_embed), torch.cos(_embed)),
            2
        ).to(steps.device)
        diff_embed = StepEmbedding.swish(self.fc_t1(diff_embed))
        diff_embed = StepEmbedding.swish(self.fc_t2(diff_embed))
        return diff_embed