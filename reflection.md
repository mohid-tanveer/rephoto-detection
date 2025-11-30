# reflection on implementation vs original plan

## high-level alignment

- **core idea preserved**: both the original plan and the final implementation center on a **hybrid detector** that fuses three decision signals:
  - moire cues,
  - subpixel color-structure cues,
  - an exif-based prior.
- **evaluation protocol preserved**: the implementation follows the planned evaluation:
  - base train/test metrics on a held-out test set,
  - **leave-one-display-type-out (lod)** and **leave-one-camera-out (loc)** splits to probe generalization.

## what changed in the implementation

- **moire modeling upgraded to a wavelet + cnn branch**:
  - the original plan envisioned moire as a **handcrafted feature + classical classifier** (e.g., spectral net on top of moire/subpixel features).
  - in the final system:
    - the moire decision signal comes from a **dual-branch wavelet cnn** (`MoireWaveletModel` in `src/models/moire_wavelet_cnn.py`) trained on:
      - wavelet coefficient stacks,
      - spatial rgb tensors.
  - this shift to a cnn was motivated by:
    - the desire to let the model discover richer, non-linear moire patterns than hand-coded statistics could capture,
    - early iterations showing that classical models on high-dimensional moire features were fragile and tedious to tune.
    - high values for fpr@95 tpr and low auc values on the original moire signal.

- **subpixel model simplified to a logistic head on richer features**:
  - the plan called for a **spectral net** or small mlp on concatenated moire + subpixel features.
  - we initially prototyped a neural classifier, but opted for a simpler, more stable logistic regression when the dataset size and feature dimensionality made heavier models hard to justify
  - in the final code:  
    - subpixel cues are extracted via an extended version of the planned algorithm (cross power between channels, period consistency, edge strength, and summary stats) in `src/features/subpixel.py`,
    - they feed a **`LogisticHead`** (`src/models/logistic_head.py`) rather than the heavier spectral network.

- **exif prior realized as a strong random forest baseline**:
  - the plan already specified a random forest on engineered exif features as a low-cost prior.
  - the final `ExifPriorModel` largely matches that, but now:
    - exif feature engineering is more systematic (`engineer_features` in `src/features/exif.py`),
    - caching and alignment with the index table are fully integrated into the pipeline.
  - in practice, this exif prior turned out to be **much stronger than expected**, often matching or exceeding the hybrid performance on all splits.

- **fusion implemented as a torch linear head with early stopping**:
  - the plan proposed a logistic regression stacking layer on `[p_moire, p_subpixel, p_exif]`.
  - the implemented `HybridFusionModel` uses a **single-layer torch network** with:
    - `BCEWithLogitsLoss` and adam,
    - early stopping monitored on a validation split.
  - functionally this is very close to logistic regression, but the torch implementation made it easier to share training infrastructure with the wavelet cnn and to extend the fusion head if needed.

## what the metrics tell us

the notebook reports metrics at three levels: overall train/test, lod (by `screen_type`), and loc (by `camera_body`). across all three views, the behavior of each signal is very different from the initial intuition.

- **overall train/test performance**:
  - **moire**:
    - training and test aucs around **0.98+**, but **fpr@95 tpr** roughly in the **0.07–0.15** range.
    - interpretation: moire is a meaningful signal, but not clean enough to stand alone at low-false-positive operating points.
  - **subpixel**:
    - aucs roughly **0.84–0.89**, but **fpr@95 tpr** close to **0.5** on both train and test.
    - interpretation: the model ranks many positive examples reasonably well, but the score distribution for negatives heavily overlaps with positives; at realistic tpr levels it produces **too many false positives**.
  - **exif**:
    - auc **1.0** and **fpr@95 tpr = 0.0** on both train and test.
    - interpretation: within this dataset, exif alone almost perfectly separates re-photos from authentic images.
  - **hybrid**:
    - mirrors exif with auc **1.0** and **fpr@95 tpr = 0.0**, suggesting:
      - the fusion head has effectively learned to **trust exif almost completely** and treat moire/subpixel as weak or noisy auxiliaries.

- **leave-one-display-type-out (lod)**:
  - for the held-out **lcd** and **oled** cases:
    - **moire**:
      - auc in the **0.82–0.87** range,
      - **fpr@95 tpr** around **0.4–0.67**.
      - decent ranking but high false positive rates once we push tpr high.
    - **subpixel**:
      - auc in the **0.72–0.77** range,
      - **fpr@95 tpr** often approaching **1.0**.
      - effectively unusable at high sensitivity; it tends to call far too many authentic images “re-photos.”
    - **exif**:
      - auc **1.0**, **fpr@95 tpr = 0.0** even when we remove all re-photos of a given display type from training.
      - exif generalizes almost perfectly within the limited distribution of this dataset.
    - **hybrid**:
      - auc near **1.0** with very low **fpr@95 tpr** (close to 0).
      - again, effectively riding on the exif signal.

- **leave-one-camera-out (loc)**:
  - across iPhone 12 Pro Max, 14 Pro Max, and 17 Pro:
    - **moire**:
      - aucs in the **0.75–0.91** range,
      - **fpr@95 tpr** around **0.3–0.5**.
      - similar story: somewhat helpful but not reliable at strict operating points.
    - **subpixel**:
      - aucs vary widely (**0.65–0.90**), but **fpr@95 tpr** is often **very high (up to ~1.0)**.
      - indicates strong **false positive tendencies** when the camera configuration changes.
    - **exif**:
      - auc **1.0**, **fpr@95 tpr = 0.0** across cameras.
    - **hybrid**:
      - again near-perfect, tracking exif.

these results clearly show that:

- the **exif prior is dominating**, providing nearly perfect separation and generalization within this dataset,
- **moire** is a useful but secondary cue,
- **subpixel**, at least in this implementation and dataset, is **not a reliable indicator of re-photos**.

## interpreting subpixel’s poor performance

from the metrics and the feature design, several plausible explanations emerge for why subpixel detection underperforms and produces so many false positives:

- **screen and camera pipelines blur subpixel structure**:
  - modern displays (especially high-density oled and lcd) and modern phone cameras aggressively apply:
    - subpixel rendering,
    - temporal dithering,
    - demosaicing,
    - denoising, sharpening, and resizing.
  - by the time an image is saved as a jpeg and possibly rescaled, the crisp subpixel grid we expect conceptually may be **partially smeared into generic high-frequency texture**, which is also present in many authentic scenes (e.g., fabric, foliage, building facades).

- **feature design may be capturing generic high-frequency structure, not screen-specific cues**:
  - the cross-power peaks and period measures respond strongly to **any** strong regular structure or high-contrast edges.
  - if re-photos tend to be of relatively flat, synthetic content (ui, illustrations) and authentic images include lots of textured natural scenes, it is easy for the subpixel features to confuse:
    - fine edges and repeating patterns in real scenes,
    - with genuine screen subpixel patterns.
  - this would explain why auc is non-trivial (it sometimes ranks positives higher) but **fpr explodes** at high tpr.

- **limited dataset vs. high-dimensional feature space**:
  - the subpixel summary includes many different statistics (ratios, periods, percentiles, consistency measures).
  - with only a few hundred images, logistic regression can easily learn spurious correlations that work on the training set but do not generalize to new cameras or display types.
  - the lod/loc metrics suggest exactly this: what looked decent in-sample does not transfer.

- **tile scale and alignment may be suboptimal**:
  - tiles of size 256–512 pixels at the final image resolution may not align nicely with the underlying subpixel grid.
  - any misalignment, plus the use of relatively large tiles, can average out subpixel-level information and make the cross-spectral cues much noisier.
  - after testing, however, we found that the auc and fpr@95 tpr values did not change significantly when we changed the tile size or stride.

in short, the subpixel branch currently behaves more like a **generic “high-frequency texture detector”** than a robust screen-specific cue.

## interpreting exif’s very strong performance

exif’s near-perfect performance is both encouraging and suspicious:

- **why it could genuinely be strong**:
  - in many real-world workflows, re-photos are captured with:
    - relatively short focus distances,
    - particular aperture and shutter combinations (especially indoors),
    - distinctive iso and metering choices,
  - authentic images, especially varied real-world scenes, can have much more diverse and often distinctly different exif profiles.
  - if those patterns hold broadly, an exif prior could indeed be a very strong signal.

- **why it might be overestimating generalization**:
  - in this project, **we captured the re-photos in a fairly consistent way**:
    - similar devices (mostly iphones),
    - similar zoom levels and focal lengths,
    - similar display types and environments.
  - that makes it easier for the random forest to latch onto dataset-specific quirks such as:
    - exact combinations of focal length, iso, and shutter speed
    - particular exposure and metering modes used for re-photos vs. authentic scenes.
  - because these quirks are consistent across the lod/loc splits, exif appears to generalize, but that is **within the same overall acquisition protocol**.

the correct reading is probably:

- **within this dataset**, exif is an extremely strong and stable cue,
- but we **cannot yet claim** it is robust to:
  - different operators and capture styles

that uncertainty is important to call out as a limitation and as a clear target for future work.

## interpreting moire’s moderate performance

moire performs better than subpixel but clearly lags behind exif:

- **why it helps**:
  - for images where the screen and camera geometry create visible moire patterns, the moire features (and the wavelet cnn) can reliably separate re-photos from authentic images.
  - the relatively high auc values suggest that a subset of re-photos carries distinctive frequency-domain signatures that the model can exploit.

- **why it struggles on lod/loc**:
  - moire is highly sensitive to:
    - screen pixel pitch and subpixel layout,
    - capture distance and angle,
    - focus and motion blur,
    - image content (e.g., high-frequency line art vs. smooth gradients).
  - our dataset consisted of re-photos that attempted to mask the tells of re-photos (moire, subpixel, etc.), so the moire model may have been set up to fail (well not work as well as it could).
  - when you hold out an entire screen type or camera body, those geometric and optical conditions change enough that the learned moire patterns only partially transfer.

so moire is a **valuable secondary cue**, but on this dataset it cannot match exif’s reliability, and it alone would not support low-false-positive deployment.

### future directions and open questions

based on the plan vs. implementation and the observed metrics, several follow-up steps seem especially important.

- **stress-test exif generalization**
  - collect additional data where:
    - re-photos are taken on **different devices**, at **different focal lengths**, and with **varied capture settings** (but would need to capture re-photos at the same quality of those in this dataset)
    - authentic images share similar exif profiles (e.g., same device and similar exposure settings but with no screen present).
  - re-run the full pipeline on this expanded dataset to see:
    - whether exif alone still scores near-perfect,
    - or whether performance degrades when the protocol-specific shortcuts are removed.

- **revisit subpixel detection**
  - experiment with:
    - possibly **learning subpixel cues with a small cnn** directly on high-resolution crops around edges, rather than relying solely on handcrafted spectra.
  - in parallel, perform targeted qualitative analysis:
    - visualize cross-power spectra and tile-level scores on a few representative re-photos and authentic images to see what the current features are actually picking up.

- **strengthen and interpret moire modeling**
  - augment the dataset with scenes and displays that are more likely to produce moire:
    - high-frequency content on screens (text, striped patterns),
    - varied viewing angles and distances.
  - use the existing wavelet cnn but:
    - add intermediate feature visualizations or saliency maps to understand which regions drive its predictions,
    - test whether combining engineered moire statistics with the cnn embeddings improves robustness.

## concluding reflection

the final implementation faithfully realizes the original concept of a hybrid re-photo detector, but the **relative importance of the three signals** is quite different from what we initially expected: exif dominates, moire helps but is secondary, and the current subpixel branch is not yet reliable. this is a useful outcome in itself: it highlights that **capture metadata can be extremely informative in controlled settings**, but also that we need more diverse data and more careful modeling if we want robust, image-based cues that generalize beyond a single capture protocol. for future iterations, the most impactful work is likely to be **testing exif’s limits** on new datasets and **substantially rethinking subpixel modeling**, treating the current implementation as a first, informative prototype rather than a final answer.
