import os
import paddle
import argparse
import paddle.vision.transforms as transforms

from tqdm import tqdm
from model import CheXNet
from paddle.io import DataLoader
from data import ChestXrayDataSet
from utility import N_CLASSES, CLASS_NAMES, TenCrop, Lambda, AUROC

def print_aurocs(AUROCs):
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROCs[-1]))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

def evaluate(args):
    # initialize and load the model
    model = CheXNet(N_CLASSES)

    if os.path.isfile(args.ckpt):
        print("=> loading checkpoint")
        checkpoint = paddle.load(args.ckpt)
        model.set_dict(checkpoint)
        print("=> loaded checkpoint")
    else:
        raise ValueError("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(
        data_dir=args.data_dir,
        image_list_file=args.test_list,
        transform=transforms.Compose([
            transforms.Resize(256),
            TenCrop(224),
            Lambda(lambda crops: paddle.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            Lambda(lambda crops: paddle.stack(
                [normalize(crop) for crop in crops]))
        ]))
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0)

    auroc = AUROC(num_classes=N_CLASSES, class_names=CLASS_NAMES)
    # # initialize the ground truth and output tensor

    # switch to evaluate mode
    model.eval()
    with paddle.no_grad():
        for i, (inp, target) in enumerate(tqdm(test_loader)):
            target = target
            bs, n_crops, c, h, w = inp.shape
            input_var = paddle.to_tensor(inp.reshape((-1, c, h, w)))
            output = model(input_var)
            output_mean = output.reshape((bs, n_crops, -1)).mean(1)
            auroc.update(output_mean, target)

    AUROCs = auroc.accumulate()
    print('The final AUROCs: ')
    print_aurocs(AUROCs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ChestX-ray14/images')
    parser.add_argument('--test_list', type=str, default='ChestX-ray14/labels/test_list.txt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ckpt', type=str, default='pretrained_models/best_model_via_this_project.pdparams')
    args = parser.parse_args()

    evaluate(args)
