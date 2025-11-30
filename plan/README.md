# Re-photo Detection Research Plan

## Goal + Background

- Develop a verifier to defend against picture-of-picture (re-photo) attacks on cryptographically signed images.
- C2PA - technical standard designed to help verify the origin and integrity of digital media (e.g. images).
  - Embeds provenance metadata that answers:
    - Who created the content (person, organization, or device).
    - When and where it was created.
      - How it’s been edited or altered through a chain of custody (a verifiable record of all modifications).
- What it doesn’t defend against however, is whether or not a physically captured image is of a screen. Which is the focus of this project.

### Motivation

- Why is this necessary?
  - Attackers today can generate/modify an image using generative models and take a picture of the image on their screen to pass off as a physically captured image.
- Use Cases:
  - Verifying authenticity of images
    - Publications
    - Courts
    - Insurance Claims
  - Protecting against misinformation
    - Social media
    - News
    - Politics
    - Elections
    - etc.

## Approach

- Dataset of ~500 labeled images:
  - ~150 authentic captures.
  - ~150 rephotos of the authentic captures.
  - ~200 rephotos of AI generated/altered images.
  - NOTE: the rephotos will also be diversified amongst different display types (mini-LED/OLED) and capture devices.

- Model architecture:
  - Hybrid approach looking at 3 decision signals.
    - Moiré patterns.
    - Subpixel cues.
    - Train model on EXIF data features (focal length, ISO, aperture, etc.) to classify if a photo was taken with these features.

### Moiré patterns

<img style="display: block; margin: 0 auto;" src="https://upload.wikimedia.org/wikipedia/commons/0/03/Moir%C3%A9_pattern.png" alt="Moiré pattern" width="300">

- Algorithm: Tile the image. For each tile, take a FFT of luminance.
- Decision signal: Look for high-frequency peaks concentrated at specific angles. Screens create stable, sharp spectral lines.

### Subpixel cues

<img style="display: block; margin: 0 auto;" src="https://upload.wikimedia.org/wikipedia/commons/5/57/Subpixel-rendering-RGB.png" alt="Subpixel cues" width="300">

- Algorithm: Compute the cross-power spectrum between R–G and G–B per tile; also compute autocorrelation in each channel.

- Decision signal: Look for peaks across channels at consistent spatial periods and period stability across neighboring tiles, and orientation consistency (grid-like patterns).

### EXIF data features

- Features: Focal length (35mm-eq), focus distance, f-number, shutter, ISO, white balance CCT, exposure/ metering modes, device model.
- Model: Train a Random Forest Classifier on labeled (rephoto vs real-scene) images.
- Acts as a low-cost prior (e.g., unusually short focus distance + low EV + high ISO indoors raises screen likelihood) and fuses with the moiré/subpixel scores.

### Evaluation

- We’ll evaluate on the test set using leave-one-display-type-out and leave-one-camera-out splits to measure generalization.
- Primary metrics are AUC and FPR at 95% TPR.
- We’ll include derivatives (EXIF-only, moiré-only, subpixel-only, etc.) and brief error analysis to show which signals drive performance and where failures occur.
