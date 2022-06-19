import numpy as np
import torch

import os
import argparse
from shutil import rmtree
from collections import defaultdict
from tqdm.auto import tqdm

from config import build_datasets, build_loaders, build_model,\
    build_criterion, build_metric_dict, build_optimizer, build_scheduler


def train_epoch(model, loader, optimizer, scheduler, loss_fn, metric_dict, device):
    model.train()
    metrics = defaultdict(list)
    for mix, src in tqdm(loader, desc='Training'):
        optimizer.zero_grad()

        mix = mix.to(device)
        src = src.to(device)

        pred = model(mix)

        loss = loss_fn(pred, src)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        metrics['loss'].append(loss.item())
        with torch.inference_mode():
            for metric, metric_func in metric_dict.items():
                if metric == 'SDR':
                    values = metric_func(pred.cpu(), src.cpu())
                else:
                    values = metric_func(pred, src)
                value = torch.mean(values)
                metrics[metric].append(value.item())
    metrics.update({f'{key}_epoch': np.mean(values) for key, values in metrics.items()})
    return metrics


@torch.inference_mode()
def test_epoch(model, loader, loss_fn, metric_dict, device):
    model.eval()
    metrics = defaultdict(list)
    for mix, src in tqdm(loader, desc='Testing'):
        mix = mix.to(device)
        src = src.to(device)

        pred = model(mix)
        loss = loss_fn(pred, src)

        metrics['loss'].append(loss.item())
        for metric, metric_func in metric_dict.items():
            if metric == 'SDR':
                values = metric_func(pred.cpu(), src.cpu())
            else:
                values = metric_func(pred, src)
            value = torch.mean(values)
            metrics[metric].append(value.item())
    metrics.update({f'{key}_epoch': np.mean(values) for key, values in metrics.items()})
    return metrics


def run(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs, metric_dict, device, saveroot):
    if os.path.exists(saveroot):
        ans = input(f'{saveroot} is already exists. Do you want to rewrite it? y/n: ')
        if ans == 'y':
            rmtree(saveroot)
        else:
            exit(1)
    os.makedirs(saveroot)

    model.to(device)
    best_loss = float('inf')
    train_metrics_run = defaultdict(list)
    test_metrics_run = defaultdict(list)
    for epoch in tqdm(range(epochs), desc='epochs'):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, metric_dict, device)
        test_metrics = test_epoch(model, test_loader, loss_fn, metric_dict, device)

        # print metrics
        print('\n')
        print(f'Finished epoch #{epoch + 1}')
        print('TRAIN:')
        for key, value in train_metrics.items():
            if 'epoch' in key:
                print(f'{key} = {value}')
        print('\nTEST:')
        for key, value in test_metrics.items():
            if 'epoch' in key:
                print(f'{key} = {value}')
        print('\n')

        # extend run metrics
        for key, value in train_metrics.items():
            if isinstance(value, list):
                train_metrics_run[key].extend(value)
            else:
                train_metrics_run[key].append(value)
        for key, value in test_metrics.items():
            if isinstance(value, list):
                test_metrics_run[key].extend(value)
            else:
                test_metrics_run[key].append(value)

        # saving last epoch
        snapshot = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        torch.save(snapshot, os.path.join(saveroot, 'last_snapshot.tar'))

        # saving best epoch based on validation loss
        cur_loss = test_metrics['loss_epoch']
        if cur_loss < best_loss:  # noqa
            torch.save(snapshot, os.path.join(saveroot, 'best_snapshot.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--cuda', type=str, required=True)
    parser.add_argument('--saveroot', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset, test_dataset = build_datasets()
    train_loader, test_loader = build_loaders(train_dataset, test_dataset)
    model = build_model(args.model_name)
    criterion = build_criterion('SI-SNR').to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, warmup_steps=2 * len(train_loader))
    metric_dict = build_metric_dict(['SNR', 'SDR', 'SI-SNR', 'SI-SDR'])

    run(model, train_loader, test_loader, optimizer, scheduler, criterion,
        args.epochs, metric_dict, device, args.saveroot)
