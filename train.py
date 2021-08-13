import math
import argparse

from model import CheXNet
from paddle import Model
from paddle.nn import BCELoss
from data import ChestXrayDataSet
from paddle.optimizer import Adam
from paddle.vision import transforms
from paddle.callbacks import EarlyStopping
from paddle.optimizer.lr import PiecewiseDecay
from utility import N_CLASSES, CLASS_NAMES, AUROC


def train(args):
    model = CheXNet(N_CLASSES)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=args.data_dir,
                                     image_list_file=args.train_list,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.RandomCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), normalize
                                     ]))

    val_dataset = ChestXrayDataSet(data_dir=args.data_dir,
                                   image_list_file=args.val_list,
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(), normalize
                                   ]))

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)

    decay_epochs = [
        int(epoch_num) for epoch_num in args.decay_epochs.split(',')
    ]
    scheduler_lr = PiecewiseDecay(
        boundaries=[epoch * steps_per_epoch for epoch in decay_epochs],
        values=[
            args.learning_rate * (args.decay_factor**i)
            for i in range(len(decay_epochs) + 1)
        ],
        last_epoch=-1,
        verbose=False)

    opt = Adam(scheduler_lr, parameters=model.parameters())

    model = Model(model)
    model.prepare(optimizer=opt,
                  loss=BCELoss(),
                  metrics=AUROC(num_classes=N_CLASSES,
                                class_names=CLASS_NAMES))

    early_stopping = EarlyStopping(monitor='AUROC_avg',
                                   mode='max',
                                   patience=10,
                                   verbose=1,
                                   min_delta=0,
                                   baseline=None,
                                   save_best_model=True)

    model.fit(train_data=train_dataset,
              eval_data=val_dataset,
              batch_size=args.batch_size,
              epochs=args.epoch,
              eval_freq=1,
              log_freq=10,
              save_dir=args.save_dir,
              save_freq=1,
              verbose=2,
              drop_last=False,
              shuffle=True,
              num_workers=0,
              callbacks=[early_stopping])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ChestX-ray14/images')
    parser.add_argument('--train_list', type=str, default='ChestX-ray14/labels/train_list.txt')
    parser.add_argument('--val_list', type=str, default='ChestX-ray14/labels/val_list.txt')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--decay_epochs', type=str, default='10, 15, 18')
    parser.add_argument('--decay_factor', type=float, default=0.1)
    args = parser.parse_args()

    train(args)
