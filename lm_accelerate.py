import os
import math
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_scheduler

import wandb
from d3pm_runner import D3PM
from dit import DDiT_Llama
from accelerate import Accelerator


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer=None, type_path="train", max_length=512, debug=False):
        if debug:
            vernum = 2
        else:
            vernum = 103
        self.vernum = vernum
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        # self.dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return (
            int(len(self.dataset) * 0.1) if (self.vernum == 103) else len(self.dataset)
        )

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # logger.info(text)
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.squeeze()
        else:
            # use byte encoding
            seq = list(text.encode("utf-8"))
            if len(seq) < self.max_length:
                seq += [0] * (self.max_length - len(seq))
            else:
                seq = seq[: self.max_length]
            input_ids = torch.tensor(seq, dtype=torch.long)

        return {"input_ids": input_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--eval_freq", type=int, default=600)
    parser.add_argument("--ckpt_freq", type=int, default=600)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="If the training should continue from a checkpoint folder.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.dump_dir) / "checkpoints"
    log_dir = Path(args.dump_dir) / "logdir"
    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="tensorboard", project_dir=args.dump_dir)
    device = accelerator.device

    N = 256
    max_length = 256
    num_train_epochs = 5

    d3pm = D3PM(
        DDiT_Llama(N, dim=512, n_layers=6), 1000, num_classes=N, hybrid_loss_coeff=0.0
    )

    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    dataset = WikiTextDataset(max_length=max_length, debug=False)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=2e-4)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optim,
        num_warmup_steps=100,
        num_training_steps=num_train_epochs * math.ceil(len(dataloader)),
    )

    d3pm, optim, dataloader, lr_scheduler = accelerator.prepare(d3pm, optim, dataloader, lr_scheduler)

    global_step = 0
    start_epoch = 0
    resume_step = None
    if args.resume:
        # Get the most recent checkpoint
        dirs = [f for f in checkpoint_dir.iterdir() if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # if a ckpt exists
        if path:
            # Extract `step_{i}`
            resume_step = int(path.name.replace("step_", ""))
            start_epoch = resume_step // len(dataloader)
            resume_step -= start_epoch * len(dataloader)
            global_step = resume_step
            accelerator.load_state(path)
            accelerator.print("loaded ctkp from: {path}")
    accelerator.print(f"step: {global_step}, epoch: {start_epoch}")

    d3pm.train()

    for i in range(start_epoch, num_train_epochs):

        if args.resume and i == start_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_step)
            global_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = dataloader

        pbar = tqdm(active_dataloader, total=len(dataloader), initial=resume_step)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x["input_ids"].to(device)

            # discritize x to N bins

            loss, info = d3pm(x)

            accelerator.backward(loss)
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )
            optim.step()
            lr_scheduler.step()
            global_step += 1

            if global_step % args.eval_freq == 1:
                d3pm.eval()

                with torch.no_grad():

                    init_noise = torch.randint(0, N, (16, max_length)).to(device)

                    outputs = d3pm.sample_with_image_sequence(
                        init_noise, None, stride=40
                    )
                    gen_outputs = []
                    total = 0
                    # back to sentence, byte to utf-8
                    for _i in range(16):
                        sent = outputs[-1][_i].cpu().tolist()
                        correctly_parsed = True
                        try:
                            sent = b"".join([bytes([i]) for i in sent]).decode("utf-8")
                        except:
                            # if there is error, just unicodec
                            correctly_parsed = False
                            sent = "".join([chr(i) for i in sent])
                        sent = (
                            f"[{_i}] Sample Correctly parsed: "
                            + str(correctly_parsed)
                            + "\n"
                            + sent
                        )
                        total += 1 if correctly_parsed else 0

                        gen_outputs.append(sent)

                    accelerator.print(sent)
                    # make a nice html to show the generated outputs
                    html_formatted = "<br>".join(gen_outputs)
                    # log text

                d3pm.train()

                if global_step % args.ckpt_freq == 1:
                    accelerator.save_state(checkpoint_dir / f"step_{global_step}")
                    print(f"Model saved at {global_step}")