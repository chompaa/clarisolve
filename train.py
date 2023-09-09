import argparse
import copy
import os
import typing

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.dataloader
import tqdm

import dataset
import models
import utils


def train(
    mode: typing.Literal["sr", "ic"],
    train_file: str,
    eval_file: str,
    output_dir: str,
    scale: int,
    learn_rate: float,
    batch_size: int,
    num_epochs: int,
    num_workers: int,
    seed: int,
) -> None:
    output_dir = os.path.join(output_dir, f"x{scale}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    optimizer = None

    if mode == "sr":
        model = models.SRCNN().to(device)

        optimizer = optim.Adam(
            [
                {"params": model.conv1.parameters()},
                {"params": model.conv2.parameters()},
                {"params": model.conv3.parameters(), "lr": learn_rate * 0.1},
            ],
            lr=learn_rate,
        )
    else:
        model = models.ICCNN().to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=learn_rate,
        )

    criterion = torch.nn.MSELoss()

    train_dataset = dataset.TrainDataset(
        train_file, input_key="inputs", label_key="labels", normalize=mode == "sr"
    )
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataset = dataset.EvalDataset(
        eval_file, input_key="inputs", label_key="labels", normalize=mode == "sr"
    )
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = utils.AverageMeter()

        with tqdm.tqdm(
            total=(len(train_dataset) - len(train_dataset) % batch_size)
        ) as progress_bar:
            progress_bar.set_description(f"epoch: {epoch}/{num_epochs - 1}")

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=f"{epoch_losses.avg:.6f}")
                progress_bar.update(len(inputs))

        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"epoch_{epoch}.pth"),
        )

        model.eval()
        epoch_psnr = utils.AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(utils.calculate_psnr(preds, labels), len(inputs))

        print(f"eval psnr: {epoch_psnr.avg:.2f}")

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print(f"best epoch: {best_epoch}, psnr: {best_psnr:.2f}")

    torch.save(best_weights, os.path.join(output_dir, "best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="sr or ic")
    parser.add_argument(
        "--train-file", type=str, required=True, help="training dataset"
    )
    parser.add_argument(
        "--eval-file", type=str, required=True, help="evaluation dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="output directory"
    )
    parser.add_argument("--scale", type=int, default=2, help="super resolution scale")
    parser.add_argument("--learn-rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--num-epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--seed", type=int, default=123, help="seed")

    args = parser.parse_args()

    if args.mode not in ["sr", "ic"]:
        raise ValueError("mode must be either sr or ic")

    train(
        args.mode,
        args.train_file,
        args.eval_file,
        args.output_dir,
        args.scale,
        args.learn_rate,
        args.batch_size,
        args.num_epochs,
        args.num_workers,
        args.seed,
    )
