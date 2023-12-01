import argparse
import copy
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.dataloader
import torchvision
import torchsummary
import tqdm

import models
import util


def train(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    output_dir: str,
    learn_rate: float,
    batch_size: int,
    checkpoint_path: str,
    end_epoch: int,
    num_workers: int,
    seed: int,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"note: using {device}")

    torch.manual_seed(seed)

    model = model.to(device)
    print(torchsummary.summary(model, (1, 256, 256)))

    optimizer = optim.Adam(
        model.parameters(),
        lr=learn_rate,
    )

    criterion = torch.nn.MSELoss().to(device)

    train_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset, batch_size=1, shuffle=False
    )

    start_epoch = 0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)

        assert isinstance(checkpoint, dict)
        assert checkpoint["epoch"] < end_epoch

        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"checkpoint loaded from epoch {checkpoint['epoch']}")

    best_epoch = 0
    best_psnr = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # hacky but pylance won't have it any other way it seems
    train_data_len = len(train_dataloader) * batch_size

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_losses = util.AverageMeter()

        with tqdm.tqdm(
            total=(train_data_len - train_data_len % batch_size)
        ) as progress_bar:
            progress_bar.set_description(f"epoch: {epoch}/{end_epoch}")

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                # clear previous gradients
                optimizer.zero_grad()
                # calculate gradients (backpropagation)
                loss.backward()
                # update weights (gradient descent)
                optimizer.step()

                progress_bar.set_postfix(loss=f"{epoch_losses.avg:.6f}")
                progress_bar.update(len(inputs))

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(output_dir, f"epoch_{epoch}.pth"),
        )

        model.eval()
        epoch_psnr = util.AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(util.calculate_psnr(preds, labels), len(inputs))

        print(f"eval psnr: {epoch_psnr.avg:.2f}")

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print(f"best epoch: {best_epoch}, psnr: {best_psnr:.2f}")

    torch.save(best_weights, os.path.join(output_dir, "best.pth"))


if __name__ == "__main__":
    sr_models = models.SR_MODELS
    ic_models = models.IC_MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(sr_models.keys()) + list(ic_models.keys()),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="training dataset. must point to a h5 file for SR models and a directory for IC models.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="evaluation dataset. must point to a h5 file for SR models and a directory for IC models.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="output directory"
    )
    parser.add_argument("--learn-rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="checkpoint")
    parser.add_argument("--end-epoch", type=int, default=400, help="end epoch")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--seed", type=int, default=123, help="seed")

    args = parser.parse_args()

    model = None
    train_dataset = None
    eval_dataset = None

    if args.model in sr_models:
        model = sr_models[args.model]()

        train_dataset = util.TrainDataset(
            args.train,
            input_key="inputs",
            label_key="labels",
            normalize=True,
        )

        eval_dataset = util.EvalDataset(
            args.eval,
            input_key="inputs",
            label_key="labels",
            normalize=True,
        )
    elif args.model in ic_models:
        model = ic_models[args.model]()

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        )

        train_dataset = util.ColorDataset(args.train, train_transform)

        eval_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
            ]
        )

        eval_dataset = util.ColorDataset(args.eval, eval_transform)

    assert model is not None
    assert train_dataset is not None
    assert eval_dataset is not None

    train(
        model,
        train_dataset,
        eval_dataset,
        args.output_dir,
        args.learn_rate,
        args.batch_size,
        args.checkpoint_path,
        args.end_epoch,
        args.num_workers,
        args.seed,
    )
