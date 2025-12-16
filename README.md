# Re-photo Detection Research Project

This project attempts to implement a hybrid detector for identifying re-photos (images of screens) versus authentic photos. it combines:

- **wavelet + cnn modeling**: a dual-branch cnn over wavelet coefficients and spatial rgb content.
- **subpixel analysis**: color substructure patterns consistent with lcd/oled subpixels.
- **exif priors**: camera metadata and engineered exif features.
- **fusion head**: a small neural net that combines per-signal probabilities into a final score.

You can read the full paper here: [ScreenSense: Detecting Screen Re-photos at Verification Time](./ScreenSense_Detecting_Screen_Re_photos_at_Verification_Time.pdf)

The main entrypoint for experimentation is the notebook `notebooks/rephoto.ipynb`, which orchestrates the end-to-end pipeline implemented in `src/`.

## Directory layout

- **data/**: training and test images plus csv metadata.
  - **exif_metadata.csv**: exif + label metadata for training/validation.
  - **test/test_exif_metadata.csv**: exif + label metadata for the held-out test set.
  - **re-photo/** and **authentic/** subfolders: source images grouped by screen type and authenticity.
- **artifacts/**: cached features, indices, and trained model weights.
  - **features/**: serialized moire, subpixel, and wavelet feature tensors.
  - **models/**: trained models (`moire_wavelet.pt`, `subpixel.joblib`, `exif.joblib`, `fusion.pt`, `metadata.json`).
  - **test/**: analogous caches/models when evaluating on the test split.
- **src/**: python package containing the reusable pipeline implementation.
  - **data/indexer.py**: builds a canonical dataframe index over images and labels.
  - **features/**: feature extractors for exif, moire, wavelet, and subpixel signals.
  - **models/**: model classes for each signal and the fusion head.
  - **pipeline.py**: high-level orchestration for feature building, training, evaluation, and prediction.
  - **utils/image.py**: basic image loading and resizing helpers.
- **notebooks/rephoto.ipynb**: interactive notebook that configures and runs the hybrid stack and visualizes metrics.

## Data and indexing

The pipeline starts from csv metadata, primarily `data/exif_metadata.csv`, which includes:

- **filepath / filename**: location and id for each image.
- **label**: label string (e.g., `authentic` vs. re-photo variants).
- **screen_type / screen_source**: describes the display (lcd vs oled, ai vs authentic, etc.).
- **device_make / device_model / metering_mode / iso / exposure / f-number**: raw exif fields used for feature engineering.

`src/data/indexer.py` defines:

- **`IndexConfig`**: paths to `data_dir`, `exif_csv`, `artifacts_dir`, and the cache file name.
- **`load_or_build_index`**: builds or reloads a dataframe with:
  - **`abs_path`**: resolved absolute filepath for each image.
  - **`image_id`**: a stable id derived from `filename`.
  - **`label_binary`**: numeric label where 0 = authentic, 1 = re-photo.
  - **`screen_group`**: combined `screen_type` + `screen_source`.
- **`camera_body`**: combined `device_make` + `device_model`.

This index is the backbone for all downstream feature extraction and model training.

## Feature extraction

### Exif features

`src/features/exif.py` builds a dense exif-based feature table.

- **numeric columns**: `image_width`, `image_height`, `focal_length_35mm_eq`, `f_number`, `exposure_time`, `shutter_speed_value`, `iso`.
- **categorical columns**: `device_make`, `device_model`, `screen_type`, `screen_source`, `metering_mode`.
- **engineered features** include:
  - **aspect_ratio**: width / height.
  - **log_iso**: \(\log(1 + \text{iso})\).
  - **exposure_value**: brightness-related term from aperture and exposure time.
  - **focal_per_aperture**: focal length relative to aperture.
- categoricals are one-hot encoded, then concatenated with numeric features.
- **`build_exif_feature_table`** returns both the feature dataframe and a sorted list of exif feature column names.

`src/pipeline.py` also defines `_extract_raw_exif` and helpers to build exif vectors for arbitrary jpegs and cache them, enabling on-the-fly metadata-based inference outside the training set.

### Subpixel (color structure) features

`src/features/subpixel.py` looks at cross-channel periodicity consistent with subpixel layouts.

- rgb tiles are obtained via `sliding_windows` (with `tile_size`, `stride`, and `max_tiles`).
- for each tile:
  - a 2d hann window is applied to each channel.
  - 2d ffts of r, g, and b are computed.
  - cross-power spectra between channel pairs (r–g, g–b, r–b) are analyzed using the radius grid from `center_frequency_grid`.
- tile-level metrics (`SubpixelTileMetrics`) include:
  - **rg_peak_ratio / gb_peak_ratio / rb_peak_ratio**: peak cross-power vs. mean, per channel pair.
  - **rg_period / gb_period / rb_period**: dominant radial frequency for each pair.
  - **period_consistency**: inverse of period standard deviation across pairs.
  - **edge_strength**: sobel-based edge magnitude estimate.
  - a composite **score** combining peak ratios, period consistency, and edge strength.
- `extract_subpixel_features` aggregates these into:
  - distribution summaries (`mean`, `std`, `max`, percentiles) for scores, edges, consistency, ratios, and periods.
  - additional top-k and global statistics (e.g., fraction of tiles with high peak ratios).

These features are designed to pick up on repeating color subpixel grids typical of photographed screens.

### Wavelet + spatial tensors

`src/features/moire_wavelet.py` builds tensors consumed by the cnn:

- `compute_wavelet_and_spatial`:
  - loads a grayscale image and resizes it to a fixed `wavelet_size`.
  - computes a two-level 2d discrete wavelet transform (default `db2`), producing approximation and detail coefficients at two scales.
  - normalizes and stacks these into a multi-channel wavelet tensor.
  - builds a corresponding spatial rgb tensor by resizing the original rgb image to `spatial_size` and arranging it as `(channels, height, width)`.
- `build_wavelet_dataset`:
  - iterates over the index dataframe’s `abs_path` values.
  - constructs aligned arrays of wavelet and spatial tensors for all images.
  - persists them to a compressed `.npz` file in `artifacts/features`.
  - reloads them from cache when possible to avoid recomputation.

These tensors serve as the high-capacity input to the moire wavelet cnn.

## Models

### Exif prior (random forest)

`src/models/exif_prior.py` implements `ExifPriorModel`:

- wraps a `RandomForestClassifier` over engineered exif features.
- configuration (`ExifPriorConfig`) controls tree count, depth, and regularization.
- supports `fit`, `predict_proba`, and `save`/`load` via joblib, packaging both the model and the feature name ordering.

This model captures metadata-only priors about which camera and capture settings are more likely to correspond to re-photos.

### Subpixel head (logistic regression)

`src/models/logistic_head.py` implements `LogisticHead`:

- applies a `StandardScaler` followed by `LogisticRegression` in a sklearn `Pipeline`.
- `LogisticConfig` exposes regularization strength, max iterations, and class weights.
- provides `fit`, `predict_proba`, and `save`/`load` while retaining the feature name order.

This head learns a linear decision boundary over the rich subpixel statistics.

### Moire wavelet cnn

`src/models/moire_wavelet_cnn.py` defines a dual-branch cnn:

- **wavelet branch**:
  - stacked convolutional blocks with batch normalization, relu, and max pooling.
  - global average pooling to obtain a compact representation of wavelet coefficients.
- **spatial branch**:
  - similar conv blocks over rgb spatial content, with an additional higher-capacity block.
  - global average pooling to summarize spatial features.
- the two branches are concatenated and fed to a small mlp classifier.
-
- `MoireWaveletModel`:
  - manages device placement (cpu / cuda / mps).
  - trains with `BCEWithLogitsLoss` and adam, using early stopping based on validation loss.
  - exposes `predict_proba` and `save`/`load` methods.

This cnn specializes in high-frequency artifacts and structured moire patterns that may not be linearly separable.

### Hybrid fusion head

`src/models/hybrid_classifier.py` implements the fusion model:

- `HybridFusionModel` wraps a single-layer linear head over three inputs:
  - `moire_prob` (cnn-based),
  - `subpixel_prob` (logistic head),
  - `exif_prob` (random forest).
- trained with `BCEWithLogitsLoss` and early stopping on validation loss (`HybridConfig` controls lr, weight decay, epochs, patience, batch size).
- returns a final **hybrid probability** via a sigmoid over the fused logits.

This model learns how to weight each signal depending on its reliability for a given image population.

## Pipeline orchestration

`src/pipeline.py` connects indexing, feature extraction, and modeling into a single workflow.

### Configuration and feature store

- **`PipelineConfig`** defines:
  - paths: `data_dir`, `exif_csv`, `artifacts_dir`, optional `test_data_dir` and `test_exif_csv`.
  - tiling parameters: `tile_size`, `tile_stride`, `max_tiles_per_image`.
  - flags: `force_feature_recompute`, `force_index_recompute`.
  - `device`: desired compute device for torch models.
- **`FeatureStore`** holds:
  - a master `table` with labels, paths, and all engineered features.
  - `feature_groups`: column lists for `moire`, `subpixel`, and `exif` features.
  - `moire_wavelet` and `moire_spatial` tensors.
  - a `subset` method to keep all views/tensors aligned when filtering rows.

### Building features

`build_feature_store(config)` performs:

1. **index loading**: uses `IndexConfig` and `load_or_build_index` to obtain the image index dataframe.
2. **moire features**:
   - computes or reloads `moire_t{tile_size}_s{tile_stride}.pkl` in `artifacts/features`.
   - for each indexed image, loads a grayscale version and applies `extract_moire_features`.
3. **subpixel features**:
   - computes or reloads `subpixel_t{tile_size}_s{tile_stride}.pkl`.
   - loads rgb images and applies `extract_subpixel_features`.
4. **wavelet + spatial tensors**:
   - computes or reloads `wavelet_t{tile_size}.npz` via `build_wavelet_dataset`.
5. **exif features**:
   - calls `build_exif_feature_table` on the index dataframe.
6. **merge & finalize**:
   - merges moire, subpixel, and exif features with the index on `image_id`.
   - fills missing values with zeros and records feature column groups.

The result is a fully materialized `FeatureStore` ready for training and evaluation.

### Training individual models and fusion

`train_models(store, config)`:

1. extracts the binary label vector from `store.table["label_binary"]`.
2. builds numpy arrays for:
   - `moire_wavelet` and `moire_spatial` from tensors,
   - subpixel features (columns starting with `subpixel_`),
   - exif features (as returned by `build_exif_feature_table`).
3. defines model factories for:
   - `MoireWaveletModel` (cnn),
   - `LogisticHead` for subpixel,
   - `ExifPriorModel` for exif.
4. trains each signal model with stratified k-fold cross validation:
   - `_train_wavelet_model` handles wavelet+spatial, producing out-of-fold (`oof`) probabilities.
   - `_train_signal_model` handles subpixel and exif, also producing `oof` probabilities.
5. stacks the `oof` probabilities into a 3-column matrix and trains `HybridFusionModel` on this representation, with a held-out validation split for early stopping.
6. optionally saves:
   - all models to `artifacts/models`,
   - metadata (feature groups and tiling parameters) to `metadata.json`.

The function returns a `ModelBundle` containing all trained components and feature group metadata.

### Prediction and metrics

- **`predict_with_bundle(store, bundle)`**:
  - computes `moire_prob`, `subpixel_prob`, `exif_prob`, and `hybrid_prob` for all rows in `store.table`.
  - returns a dataframe with labels, screen/camera attributes, paths, and all four probability columns.
- **`summarize_metrics(pred_df)`**:
  - for each of the four probability columns:
    - computes roc auc (`auc`) against `label_binary`.
    - computes `fpr_at_95_tpr` via `roc_curve`, i.e., the false positive rate when the true positive rate is near 0.95.
  - returns a compact metrics table used in the notebook.

### Leave-one-group-out evaluation

`evaluate_leave_one(store, config, split_column)` analyzes generalization across domains:

- for each unique value in `split_column` (e.g., `screen_type` or `camera_body`):
  - builds a **train** `FeatureStore` that excludes that value, and a **test** `FeatureStore` containing only that value.
  - for `screen_type`, it constructs a test set of:
    - all re-photos of the held-out screen type, plus
    - a random subset of authentic images, ensuring both classes are present.
- retrains the full hybrid pipeline on the training subset and evaluates on the held-out subset.
- aggregates per-signal metrics along with the held-out group identifier.

This provides a robust measure of how well the detector transfers to unseen screens or camera bodies.

### Test-set evaluation

`evaluate_on_test_set(config, bundle)`:

- builds a `FeatureStore` over the designated test folders and test exif csv (often `data/test/test_exif_metadata.csv`).
- reuses the already-trained `ModelBundle` (including the fusion head).
- computes metrics via `summarize_metrics` and returns both the test feature store and its metrics table.

This is what the notebook uses for the final held-out evaluation.

### High-level helpers

- **`run_full_pipeline(config)`**:
  - builds the training feature store,
  - trains all models and fusion,
  - this is the primary function called in `notebooks/rephoto.ipynb`.
- **`build_test_store(config)`**:
  - convenience function to construct a `FeatureStore` for test data using test-specific directories while sharing the main artifacts root.
- **`build_exif_vector`**:
  - extracts an exif feature vector for a specific `image_id` from a populated `FeatureStore`, which is helpful for targeted analysis or debugging.

## Notebook workflow

`notebooks/rephoto.ipynb` wires the pipeline together in an interactive environment:

- defines `PROJECT_ROOT` and imports:
  - `PipelineConfig`, `run_full_pipeline`, `evaluate_leave_one`,
  - `predict_with_bundle`, `summarize_metrics`, `evaluate_on_test_set`.
- constructs a `PipelineConfig` that points to:
  - `data/` and `data/exif_metadata.csv`,
  - `artifacts/` for caching features and models,
  - `data/test/` and `data/test/test_exif_metadata.csv` for the held-out test set,
  - chosen tiling parameters and device (e.g., `"mps"` on mac).
- runs:
  - `store, bundle = run_full_pipeline(config)`,
  - summarizes metrics on the training data,
  - evaluates on the test set with `evaluate_on_test_set`.
- performs **leave-one-display-type-out** and **leave-one-camera-out** evaluations using `evaluate_leave_one`, displaying summary tables for each signal and the hybrid model.
- demonstrates **per-image inspection** on the test set by:
  - running `predict_with_bundle(test_store, bundle)`,
  - filtering by filename,
  - inspecting `label_binary`, `moire_prob`, `subpixel_prob`, `exif_prob`, and `hybrid_prob` for a specific image.

This notebook is the main artifact for exploring model behavior and validating the detector qualitatively and quantitatively.

## Running the project

### Environment setup

- **create a virtual environment** (example with python 3.11+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- ensure `data/` and `artifacts/` follow the expected layout (as in this repository). the provided files are already organized accordingly.

### running the notebook pipeline

- from the project root:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=rephoto-env
jupyter notebook
```

- open `notebooks/rephoto.ipynb`, select the `rephoto-env` (or `.venv`) kernel, and run the cells top to bottom:
  - first cells set up paths and imports.
  - configuration cells set tiling parameters and device.
  - pipeline cells compute features, train models, and print metrics.
  - evaluation cells run leave-one-out and test-set analyses.

### interpreting outputs

- **training metrics** table: shows auc and `fpr_at_95_tpr` for each signal (moire, subpixel, exif) and the hybrid fusion on the training set.
- **test metrics** table: same metrics on the held-out test set; this is the main benchmark for generalization.
- **leave-one-out tables**:
  - rows correspond to individual screen types or camera bodies.
  - comparing per-signal vs. hybrid performance reveals which cues generalize better across domains that are not present in the training set.
- **per-test-image metrics**:
  - the notebook’s final section shows the probabilities generated by the models for each image in the test set.
