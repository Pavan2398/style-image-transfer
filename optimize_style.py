"""
Simple Style Transfer using Optimization (Gatys et al.)
Works without pretrained weights - uses torchvision VGG19
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_image(path, size=512):
    Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return transform(img).unsqueeze(0)


def denorm(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def get_gram(features):
    b, c, h, w = features.shape
    features = features.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)


def get_features(image, vgg, layers):
    features = []
    x = image
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features


def style_transfer(content_path, style_path, output_path, 
                  iterations=500, content_weight=1.0, style_weight=1e5,
                  image_size=512, device='cuda'):
    
    print(f"Loading VGG19 on {device}...")
    vgg = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    
    # VGG layer indices for relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
    style_layers = [1, 6, 11, 20, 29]  # 0-indexed
    content_layer = [21]  # relu4_2
    
    print(f"Loading images...")
    content_img = load_image(content_path, image_size).to(device)
    style_img = load_image(style_path, image_size).to(device)
    
    # Extract features
    print("Extracting features...")
    content_features = get_features(content_img, vgg, content_layer)
    style_features = get_features(style_img, vgg, style_layers)
    
    # Compute style gram matrices
    style_grams = [get_gram(f.detach()) for f in style_features]
    
    # Initialize output as content + noise
    output = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([output], lr=0.1)
    
    print(f"Stylizing for {iterations} iterations...")
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Extract output features
        out_content = get_features(output, vgg, content_layer)
        out_style = get_features(output, vgg, style_layers)
        
        # Content loss
        content_loss = content_weight * nn.MSELoss()(out_content[0], content_features[0])
        
        # Style loss
        style_loss = 0
        for j, feat in enumerate(out_style):
            out_gram = get_gram(feat)
            style_loss += style_weight * nn.MSELoss()(out_gram, style_grams[j])
        
        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}/{iterations}: Loss={total_loss.item():.2f}, Content={content_loss.item():.2f}, Style={style_loss.item():.2e}")
    
    # Save output
    print(f"Saving to {output_path}...")
    out_img = denorm(output.detach()).squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = (out_img * 255).astype(np.uint8)
    Image.fromarray(out_img).save(output_path, quality=95)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Style Transfer (Optimization)')
    parser.add_argument('--content', type=str, required=True, help='Content image path')
    parser.add_argument('--style', type=str, required=True, help='Style image path')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output path')
    parser.add_argument('--iterations', type=int, default=500, help='Iterations')
    parser.add_argument('--content-weight', type=float, default=1.0, help='Content weight')
    parser.add_argument('--style-weight', type=float, default=1e5, help='Style weight')
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    style_transfer(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        iterations=args.iterations,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        image_size=args.image_size,
        device=args.device
    )


if __name__ == '__main__':
    main()