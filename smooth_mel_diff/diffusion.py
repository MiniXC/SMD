import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

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

    def __call__(self, c, N=4, bs=1, no_grad=True, single_element=False):
        if N not in self.noise_schedules:
            raise ValueError(f"Invalid noise schedule length {N}")

        noise_schedule = self.noise_schedules[N]

        if not isinstance(noise_schedule, torch.Tensor):
            noise_schedule = torch.FloatTensor(noise_schedule)

        noise_schedule = noise_schedule.to(c.device)
        noise_schedule = noise_schedule.to(torch.float32)

        pred = self.sampling_given_noise_schedule(
            noise_schedule,
            c,
            no_grad=no_grad,
            single_element=single_element,
        )

        return pred

    def sampling_given_noise_schedule(
        self,
        noise_schedule,
        c,
        no_grad=True,
        single_element=False,
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

        batch_size = c.shape[0]
        if single_element:
            sequence_length = 1
        else:
            sequence_length = c.shape[1]

        _dh = self.diff_params
        T, alpha = _dh["T"], _dh["alpha"]
        assert len(alpha) == T

        N = len(noise_schedule)
        beta_infer = noise_schedule
        alpha_infer = [1 - float(x) for x in beta_infer] 
        sigma_infer = beta_infer + 0

        for n in range(1, N):
            alpha_infer[n] *= alpha_infer[n - 1]
            sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
        alpha_infer = torch.FloatTensor([np.sqrt(x) for x in alpha_infer])
        sigma_infer = torch.sqrt(sigma_infer)

        # Mapping noise scales to time steps
        steps_infer = []
        for n in range(N):
            step = self.map_noise_scale_to_time_step(alpha_infer[n], alpha)
            if step >= 0:
                steps_infer.append(step)
        steps_infer = torch.FloatTensor(steps_infer)

        N = len(steps_infer)

        x = torch.normal(0, 1, size=(batch_size, sequence_length, 80)).to(device=c.device, dtype=c.dtype)
        if sequence_length == 1:
            x = x.squeeze(1)

        def sampling_loop(x, length, progress_bar=True):
            if progress_bar:
                N = tqdm(range(length - 1, -1, -1), desc="Diffusion sampling")
            else:
                N = range(length - 1, -1, -1)
            for n in N:
                step = (steps_infer[n] * torch.ones((batch_size, 1, 1))).to(device=c.device, dtype=c.dtype)
                e = self.model.forward(c, x, step)
                if sequence_length == 1:
                    e = (e.squeeze(1), None)
                x = x - (beta_infer[n] / torch.sqrt(1 - alpha_infer[n] ** 2.) * e)
                x = x / torch.sqrt(1 - beta_infer[n])
                if n > 0:
                    x = x + sigma_infer[n] * torch.normal(0, 1, size=x.shape).to(device=c.device, dtype=c.dtype)

        if no_grad:
            with torch.no_grad():
                sampling_loop(x, N)
        else:
            sampling_loop(x, N, progress_bar=False)

        return x

    def map_noise_scale_to_time_step(self, alpha_infer, alpha):
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
    diff_params = {"T": len(beta), "alpha": alpha}
    return diff_params

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