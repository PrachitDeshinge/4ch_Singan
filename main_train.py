import os
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import torch

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--gpu_id', help='GPU ID to train')
    opt = parser.parse_args()
    
    # If input_name is not provided, pick the first valid image from input_dir
    if not opt.input_name:
        images = sorted([f for f in os.listdir(opt.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not images:
            print("No valid images found in input_dir.")
            exit(1)
        opt.input_name = images[0]
    
    opt = functions.post_config(opt)
    if opt.resume:
        print("Resuming training from checkpoint as specified by --resume flag.")
    print(f"Training on device: {opt.device}")
    if opt.device.type == 'mps':
        print("MPS backend is being used (Apple Silicon GPU)")
    opt.device = functions.get_device(opt)
    
    # Initialize model lists once for the single image
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
