# AdaIN Neural Style Transfer

A web application for **arbitrary neural style transfer** using **Adaptive Instance Normalization (AdaIN)** — based on *"Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization"* (Huang & Belongie, 2017).

Upload any **content image** and any **style image**, and the app renders the content in the chosen style in a single forward pass — no per-image optimization required.

![AdaIN Algorithm](NST_Code/adain_algo.png)

## Demo

| Content | Style | Result |
|---|---|---|
| ![content](Demo_IO_Images/i-p/i_p%20image.jpg) | ![style](Demo_IO_Images/i-p/style%201.png) | ![result](Demo_IO_Images/o-p/o_p%20style%201.jpg) |

## How It Works

Instead of the classic Gatys et al. approach (which iteratively optimizes pixel values to match content and Gram-matrix style losses — slow, taking minutes per image), this project uses **AdaIN**, which achieves style transfer through a single feed-forward pass:

1. **Encode** both the content and style images using a frozen, pretrained VGG (truncated at `relu4_1`).
2. **Align statistics**: normalize the content feature map (subtract its channel-wise mean, divide by its channel-wise std — i.e. instance normalization) and then re-scale/shift it using the **style image's** mean and std.
3. **Blend with `alpha`**: interpolate between the styled features and the original content features to control style strength (`alpha=1` → full style, `alpha=0` → original content).
4. **Decode** the resulting features back into an RGB image using a trained decoder network.

```python
def adaptive_instance_normalization(content_feat, style_feat):
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean) / content_std
    return normalized_content_feat * style_std + style_mean
```

Because style is injected purely through feature statistics (no learned, style-specific weights), the same trained decoder generalizes to **any unseen style image** at inference time — unlike earlier feed-forward NST methods that required training one network per style.

## Architecture

- **Encoder (`VGGEncoder`)** — VGG-19 layers up to `relu4_1`, loaded from pretrained weights and **frozen** (no gradient updates). Split into 4 blocks (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`) to support a multi-layer style loss during training.
- **Decoder** — A mirrored CNN that upsamples `relu4_1` features back to a full-resolution RGB image. Uses reflection padding + nearest-neighbor upsampling (instead of transposed convolutions) to avoid checkerboard artifacts. This is the **only trained component** in the pipeline.

## Training

`train.py` trains the decoder on unpaired content/style image folders:

- **Content loss**: MSE between the re-encoded generated image's `relu4_1` features and the AdaIN target.
- **Style loss**: MSE between the mean/std statistics of the generated image and the style image, summed across all 4 encoder layers (a statistical alternative to Gram-matrix style loss).
- **Total loss** = `content_weight * content_loss + style_weight * style_loss`, optimized with Adam and a decaying learning rate schedule.
- Periodically checkpoints the decoder/optimizer and saves sample output grids.

```bash
python train.py \
    --content_dir ./content_data \
    --style_dir ./style_data \
    --vgg ./vgg_normalised.pth \
    --experiment my_experiment \
    --epochs 20
```

## Web App

A Flask app (`app.py`) exposes a simple upload UI (`templates/index.html`) where users can:

- Upload a content image and a style image
- Adjust the **alpha** slider to control style intensity
- View and download the stylized result

```bash
python app.py
```

The app expects pretrained weights at:

- `NST_Code/vgg_normalised.pth` (frozen VGG encoder)
- `NST_Code/experiment/final_exp/decoder_final.pth` (trained decoder)

## Project Structure

```
AI-NST-PROJECT/
├── Demo_IO_Images/          # Example input/output pairs
├── NST_Code/
│   ├── app.py                # Flask web app (inference)
│   ├── train.py               # Decoder training script
│   ├── utils/
│   │   ├── models.py         # VGGEncoder + Decoder architectures
│   │   └── utils.py          # AdaIN, mean/std, dataset loader
│   ├── templates/index.html  # Upload UI
│   ├── content_data/         # Sample content images
│   ├── style_data/           # Sample style images
│   └── examples/             # Example outputs
├── code.ipynb               # Notebook: feature/activation map visualization
├── requirements.txt
└── Procfile.txt              # Gunicorn entry point for deployment
```

## Installation

```bash
git clone https://github.com/maithanirahul095-code/AI-NST-PROJECT.git
cd AI-NST-PROJECT/NST_Code
pip install -r ../requirements.txt
```

### Requirements

- Flask, Flask-Bootstrap, Flask-WTF, WTForms
- PyTorch, Torchvision
- Pillow, NumPy, tqdm
- Gunicorn (for deployment)

## Deployment

The included `Procfile.txt` runs the app with Gunicorn, suitable for platforms like Heroku:

```
web: gunicorn --bind :$PORT app:app
```

## Key Design Choices

- **Frozen encoder, trained decoder** — a clean transfer-learning setup; only the decoder needs training.
- **Instance normalization over batch normalization** — style is a per-image property, so statistics are computed per-image, per-channel rather than across a batch.
- **Reflection padding + nearest-neighbor upsampling** in the decoder — avoids the checkerboard artifacts common with transposed convolutions.
- **Adjustable `alpha`** — lets users control style strength at inference time via a simple linear interpolation between content and stylized features.

## References

- Huang, X., & Belongie, S. (2017). [*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868).
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks.*
