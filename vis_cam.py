import os
import cv2
import paddle
import argparse
import numpy as np
import paddle.vision.transforms as transforms

from chexnet.model import CheXNetCAM
from chexnet.utility import N_CLASSES, CLASS_NAMES


net_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    lambda x:x[None, ...]
])


img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    lambda x: x.astype('uint8')
])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='vis_cam/images')
    parser.add_argument('--save_dir', type=str, default='vis_cam/results')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ckpt', type=str,
                        default='pretrained_models/model_paddle.pdparams')
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    img_paths = [os.path.join(args.data_dir, x)
                 for x in os.listdir(args.data_dir) if is_image_file(x)]
    save_paths = [os.path.join(args.save_dir, x)
                  for x in os.listdir(args.data_dir) if is_image_file(x)]

    input_tensor = []
    img_oris = []
    for img_path in img_paths:
        img = cv2.imdecode(np.fromfile(
            img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor.append(net_transform(img_rgb))
        img_oris.append(img_transform(img))
    input_tensor = paddle.concat(input_tensor, axis=0)
    N = input_tensor.shape[0]
    if N > args.batch_size:
        input_tensors = paddle.split(input_tensor, [
                                     args.batch_size] * (N // args.batch_size) + [N % args.batch_size], axis=0)
    else:
        input_tensors = [input_tensor]

    net = CheXNetCAM(N_CLASSES, backbone_pretrained=False)
    params = paddle.load(args.ckpt)
    net.set_dict(params)
    net.eval()
    with paddle.no_grad():
        all_results = []
        all_cams = []
        for input_tensor in input_tensors:
            results, cams = net(input_tensor)
            all_results.append(results)
            all_cams.append(cams)
        all_results = paddle.concat(all_results, axis=0).numpy()
        all_cams = paddle.concat(all_cams, axis=0).numpy()

    for img_ori, results, cams, save_path in zip(img_oris, all_results, all_cams, save_paths):
        img_cp = img_ori.copy()
        cv2.putText(img_cp, 'Original', (5, 224-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mixes = [img_cp]
        for i, cam in enumerate(cams):
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            mix = (heatmap * 0.3 + img_ori * 0.5).astype('uint8')
            cv2.putText(mix, f'{CLASS_NAMES[i]}: {results[i]*100:.02f}%', (5, 224-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            mixes.append(mix)

        a = np.concatenate(mixes[0:5], 1)
        b = np.concatenate(mixes[5:10], 1)
        c = np.concatenate(mixes[10:15], 1)
        out = np.concatenate([a, b, c], 0)

        cv2.imwrite(save_path, out)

        if args.show:
            cv2.imshow('preview', out)
            cv2.waitKey(0)
