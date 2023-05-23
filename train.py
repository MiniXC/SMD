import json
import sys
from copy import deepcopy

from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import HfArgumentParser
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from nnAudio.features.mel import MelSpectrogram
import torchaudio
import torchaudio.transforms as AT
from librosa.filters import mel as librosa_mel
import librosa
from scipy import ndimage
from collections import deque
from transformers import get_linear_schedule_with_warmup


from smooth_mel_diff import SMDiffusion, SMDScheduler
from smooth_mel_diff.utils import NoamLR
from smooth_mel_diff.diffusion import compute_diffusion_params, compute_diffusion_params_sigmoid, DiffusionSampler
from training.arguments import Args

def eval_loop(accelerator, model, eval_ds, global_step, n_steps, diff_params, fig_max_len=256):
    sampler = DiffusionSampler(model, diffusion_params=diff_params)
    loss = 0
    for i, batch in enumerate(eval_ds):
        if i == 0:
            # run diffusion
            preds = sampler(batch, N=n_steps, return_sequence=True)
            # save to file using plt.imshow
            eval_mse = 0
            for j in range(len(preds)):
                pred = preds[j].squeeze().cpu().numpy()
                cond = batch["cond_mel"].squeeze().cpu().numpy()
                real = batch["real_mel"].squeeze().cpu().numpy()
                # calculate MSE
                eval_mse += torch.nn.functional.mse_loss(torch.tensor(pred), torch.tensor(real))
                fig, axs = plt.subplots(len(pred), 3, figsize=(10, 10))
                for i, (p, r) in enumerate(zip(pred, real)):
                    axs[i, 0].imshow(cond[i][:fig_max_len].T, origin="lower")
                    axs[i, 1].imshow(p[:fig_max_len].T , origin="lower")
                    axs[i, 2].imshow(r[:fig_max_len].T, origin="lower")
                    if i == 0:
                        axs[i, 0].set_title("cond")
                        axs[i, 1].set_title("pred")
                        axs[i, 2].set_title("real")
                    axs[i, 0].axis('off')
                    axs[i, 1].axis('off')
                    axs[i, 2].axis('off')
                plt.tight_layout()
                plt.savefig(f"test_{j}.png")
                plt.close()
                wandb.log({f"eval_image_{j}": wandb.Image(f"test_{j}.png")}, step=global_step)
            # show individual steps
            del batch["real_mel"]
            output = model(
                cond_mel=batch["cond_mel"],
                noisy_mel=batch["noisy_mel"],
                step=batch["step"],
                z=batch["z"],
            )
            pred_val = output["pred_z"].squeeze().cpu().numpy()
            real_val = batch["z"].squeeze().cpu().numpy()
            noisy_mel = batch["noisy_mel"].squeeze().cpu().numpy()
            fig = plt.figure()
            eval_mse = eval_mse / len(preds)
            wandb.log({"eval_mse": eval_mse}, step=global_step)
            print(f"eval mse: {eval_mse}")
            # show every element of the batch using imshow (make sure to do so for both pred and real)
            fig, axs = plt.subplots(len(pred_val), 5, figsize=(10, 10))
           
            for i, (p, r) in enumerate(zip(pred_val, real_val)):
                step = batch['step'][i].squeeze().cpu().numpy()
                axs[i, 0].imshow(p[:fig_max_len].T, origin="lower")
                axs[i, 1].imshow(r[:fig_max_len].T, origin="lower")
                axs[i, 2].imshow(np.abs(p - r)[:fig_max_len].T, origin="lower")
                axs[i, 3].imshow(noisy_mel[i][:fig_max_len].T, origin="lower")
                alpha = sampler.diff_params["alpha"][step].squeeze().cpu().numpy()
                delta = np.sqrt(1 - alpha ** 2.)
                axs[i, 4].imshow(((noisy_mel[i] - p * delta) / alpha)[:fig_max_len].T, origin="lower")
                if i == 0:
                    axs[i, 0].set_title("pred")
                    axs[i, 1].set_title("real")
                    axs[i, 2].set_title("diff")
                    axs[i, 3].set_title("mel")
                    axs[i, 4].set_title("denoised")
                axs[i, 0].axis('off')
                axs[i, 1].axis('off')
                axs[i, 2].axis('off')
                axs[i, 3].axis('off')
                axs[i, 4].axis('off')
                # add text to each plot with the step number
                axs[i, 0].text(0.5, 0.5, f"step {step}", horizontalalignment="center", verticalalignment="center", transform=axs[i, 0].transAxes)
            plt.tight_layout()
            plt.savefig("test_steps.png")
            plt.close()
            wandb.log({"eval_steps_img": wandb.Image("test_steps.png")}, step=global_step)
        if "real_mel" in batch:
            del batch["real_mel"]
        output = model(
            cond_mel=batch["cond_mel"],
            noisy_mel=batch["noisy_mel"],
            step=batch["step"],
            z=batch["z"],
        )
        loss += output["loss"].item()
    wandb.log({"eval_loss": loss / len(eval_ds)}, step=global_step)

class MelCollator():
    def __init__(
        self,
        diffusion_params,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        f_min=0,
        f_max=8000,
        sample_rate=22050,
        max_frames=256,
        include_real=False,
        signal_scale=1.0,
        tau=50,
        use_scheduler=False,
    ):
        self.sampling_rate = sample_rate
        self.max_frames = max_frames
        self.diff_params = diffusion_params
        self.mel_spectrogram = AT.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.mel_basis = librosa_mel(
            sr=self.sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()
        self.include_real = include_real
        self.signal_scale = signal_scale
        self.use_scheduler = use_scheduler
        self.tau = tau

    @staticmethod
    def drc(x, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def collate_fn(self, batch):
        for i, row in enumerate(batch):
            sr = self.sampling_rate
            audio_path = row["audio"]
            # load audio with torch audio and then resample
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sampling_rate:
                audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
            audio = audio[0]
            audio = audio / torch.abs(audio).max()
            mel = self.mel_spectrogram(audio).unsqueeze(0)
            mel = torch.sqrt(mel[0])
            mel = torch.matmul(self.mel_basis, mel)
            mel = MelCollator.drc(mel)
            mel = mel.T
            # normalize mel
            mel = (mel - mel.mean()) / mel.std() * self.signal_scale
            if mel.shape[0] > self.max_frames:
                mel = mel[: self.max_frames]
            elif mel.shape[0] < self.max_frames:
                # pad dimension 0
                mel = torch.nn.functional.pad(
                    mel, (0, 0, 0, self.max_frames - mel.shape[0])
                )
            batch[i]["mel"] = mel
            batch[i]["cond_mel"] = torch.tensor(ndimage.median_filter(mel, size=20))
        # stack
        batch = {
            "mel": torch.stack([row["mel"] for row in batch]),
            "cond_mel": torch.stack([row["cond_mel"] for row in batch]),
        }
        batch_size = batch["mel"].shape[0]
        step = torch.randint(low=0, high=self.diff_params["T"], size=(batch_size,1,1))
        noise_scale = self.diff_params["alpha"][step]
        delta = (1 - noise_scale**2).sqrt()
        z = torch.normal(0, 1, size=batch["mel"].shape)
        batch["noisy_mel"] = noise_scale * batch["mel"] + delta * z
        batch["z"] = z
        batch["step"] = step
        if self.use_scheduler and torch.sum((self.diff_params["T"] - step) > self.tau) == len(batch["step"]):
            noise_scale_next = self.diff_params["alpha"][step + self.tau]
            b_bound = 1 - (noise_scale_next / noise_scale)**2
            delta_bound = delta ** 2
            batch["beta_next"] = b_bound
            batch["delta"] = delta_bound
        if self.include_real:
            batch["real_mel"] = batch["mel"]
        del batch["mel"]
        return batch

def main():
    parser = HfArgumentParser([Args])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    wandb.init(
        name=args.wandb_run_name,
        project=args.wandb_project,
        mode=args.wandb_mode,
    )
    wandb.config.update(args)

    if not args.bf16:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    with accelerator.main_process_first():
        libritts = load_dataset(args.dataset)

    train_ds = libritts[args.train_split]
    eval_ds = libritts[args.eval_split]

    if args.use_scheduler:
        model_scheduler = SMDScheduler(
            in_channels=80,
            out_channels=1,
            filter_size=args.scheduler_filter_size,
            kernel_size=args.scheduler_kernel_size,
            dropout=args.scheduler_dropout,
            depthwise=args.scheduler_depthwise,
            num_layers=args.scheduler_num_layers,
        )
    else:
        model_scheduler = None

    model = SMDiffusion(
        in_channels=80,
        out_channels=80,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        depthwise=args.depthwise,
        num_layers=args.num_layers,
        step_embed_dim_hidden=args.step_embed_dim_hidden,
        model_scheduler=model_scheduler,
    )

    if args.resume_from_checkpoint:
        try:
            model.load_state_dict(torch.load(args.resume_from_checkpoint), strict=True)
        except RuntimeError as e:
            if args.strict_load:
                raise e
            else:
                print("Could not load model from checkpoint. Trying without strict loading, and removing mismatched keys.")
                current_model_dict = model.state_dict()
                loaded_state_dict = torch.load(args.resume_from_checkpoint)
                new_state_dict={
                    k:v if v.size()==current_model_dict[k].size() 
                    else current_model_dict[k] 
                    for k,v 
                    in zip(current_model_dict.keys(), loaded_state_dict.values())
                }
                model.load_state_dict(new_state_dict, strict=False)

    if args.use_sigmoid_schedule:
        diff_params = compute_diffusion_params_sigmoid(
            T=args.num_steps,
            start=args.sigmoid_start,
            end=args.sigmoid_end,
            tau=args.sigmoid_tau,
        )
    else:
        diff_params = compute_diffusion_params(
            T=args.num_steps,
            beta_0=args.beta_0,
            beta_T=args.beta_T,
        )

    collator = MelCollator(
        diffusion_params=diff_params,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        signal_scale=args.signal_scale,
        use_scheduler=args.use_scheduler,
        tau=args.scheduler_tau,        
    )

    collator_val = deepcopy(collator)
    collator_val.include_real = True

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator.collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=collator_val.collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.98],
        eps=1e-8,
    )

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval_loop(accelerator, model, eval_dataloader, 0, args.num_steps_eval, diff_params)
        return

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    progress_bar = tqdm(range(num_training_steps), desc="training", disable=not accelerator.is_local_main_process)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    model.train()

    step = 0

    # queue to store the last 100 losses
    losses = deque(maxlen=100)

    if args.use_scheduler:
        scheduler_losses = deque(maxlen=100)

    if args.lr_max_epochs is not None:
        num_scheduler_steps = args.lr_max_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    else:
        num_scheduler_steps = num_training_steps // args.gradient_accumulation_steps

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_scheduler_steps,
    )

    last_mean_loss = float("inf")

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                step += 1
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.max_grad_norm)
                outputs = model(
                    cond_mel=batch["cond_mel"],
                    noisy_mel=batch["noisy_mel"],
                    step=batch["step"],
                    z=batch["z"],
                )
                loss = outputs["loss"]
                losses.append(loss)
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (
                    args.use_scheduler 
                    and step % args.scheduler_every == 0
                    and "beta_next" in batch.keys()
                    and last_mean_loss < args.scheduler_loss_threshold
                ):
                    scheduler_loss = model.scheduler(
                        noisy_mel=batch["noisy_mel"],
                        beta_next=batch["beta_next"],
                        delta=batch["delta"],
                        z=batch["z"],
                        pred_noise=outputs["pred_z"].detach(),
                    )
                    scheduler_losses.append(scheduler_loss)
                    accelerator.backward(scheduler_loss)
                if step % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()

                steps_until_logging = args.log_every - (step % args.log_every)
                ## log losses
                if step % args.log_every == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    # get mean loss over last 100 steps
                    mean_loss = sum([l.item() for l in losses]) / len(losses)
                    last_mean_loss = mean_loss
                    wandb.log({"train/loss": mean_loss, "lr": lr}, step=step)
                    print(f"step {step}: loss={mean_loss:.4f}, lr={lr:.8f}")
                    if args.use_scheduler and len(scheduler_losses) > 0:
                        mean_scheduler_loss = sum([l.item() for l in scheduler_losses if not torch.isnan(l)]) / len(scheduler_losses)
                        wandb.log({"train/scheduler_loss": mean_scheduler_loss}, step=step)
                        print(f"step {step}: scheduler_loss={mean_scheduler_loss:.4f}")
                    wandb.log({"train/global_step": step}, step=step)
                        
                ## evaluate
                if step % args.eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_loop(accelerator, model, eval_dataloader, step, args.num_steps_eval, diff_params)
                    model.train()
                ## save checkpoint
                if step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), f"{args.checkpoint_dir}/model_{step}.pt")
                    accelerator.wait_for_everyone()
                progress_bar.update(1)
                # set description
                progress_bar.set_description(f"epoch {epoch+1}/{num_epochs}")

if __name__ == "__main__":
    main()

