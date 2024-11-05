# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img

import pandas as pd
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA batch demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--input_dir', default='./koniq10k_dataset/100_images', help='path to input image directory')
    parser.add_argument('--output_csv', default='attribute_results.csv', help='path to output CSV file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    attribute_list = ['Aesthetic', 'Happy', 'Natural', 'New', 'Scary', 'Complex']
    results = []

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for image_file in tqdm(image_files, desc="Evaluating images"):
        file_path = os.path.join(args.input_dir, image_file)
        output, attributes = restoration_inference(model, file_path, return_attributes=True)
        attributes = attributes.float().detach().cpu().numpy()[0]

        result = {'image_file': image_file}
        for attr_name, attr_value in zip(attribute_list, attributes):
            result[attr_name] = attr_value
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f'Results saved to {args.output_csv}')

if __name__ == '__main__':
    main()
