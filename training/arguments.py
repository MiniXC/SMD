from dataclasses import dataclass

@dataclass
class Args:
    # data loading
    dataset: str = "cdminix/libritts-aligned"
    train_split: str = "train"
    eval_split: str = "dev"
    num_workers: int = 16
    prefetch_factor: int = 2
    # audio
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: int = 0
    f_max: int = 8000
    max_frames: int = 256
    # model
    num_layers: int = 8
    depthwise: bool = False
    filter_size: int = 512
    kernel_size: int = 9
    dropout: float = 0.25
    step_embed_dim_hidden: int = 512
    # scheduler network
    scheduler_num_layers: int = 2
    scheduler_filter_size: int = 256
    scheduler_kernel_size: int = 3
    scheduler_dropout: float = 0.1
    scheduler_depthwise: bool = False
    use_scheduler: bool = True
    scheduler_every: int = 10
    scheduler_tau: int = 50
    scheduler_loss_threshold: float = 0.25
    # diffusion
    num_steps: int = 1000
    num_steps_eval: int = 8
    beta_0: float = 0.000001
    beta_T: float = 0.01
    use_sigmoid_schedule: bool = False
    sigmoid_start: int = 0
    sigmoid_end: int = 3
    sigmoid_tau: float = 0.5
    signal_scale: float = 1.0
    # training
    max_epochs: int = 1
    lr_max_epochs: int = None
    learning_rate: float = 2e-4
    warmup_steps: int = 1
    weight_decay: float = 0.01
    log_every: int = 500
    eval_every: int = 5000
    save_every: int = 20000
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 8
    eval_batch_size: int = 2
    gradient_sync_every: int = 100
    bf16: bool = False
    resume_from_checkpoint: str = None
    strict_load: bool = False
    max_grad_norm: float = 1.0
    train_loss_logging_sum_steps: int = 100
    gradient_accumulation_steps: int = 32
    eval_only: bool = False
    # wandb
    wandb_project: str = "smooth_mel_diffusion"
    wandb_run_name: str = None
    wandb_mode: str = "online"