# Rice Grain Instance Segmentation Pipeline

> **Objective:** Transform a raw scanned image of rice grains into a colorized instance-segmentation image where each grain is uniquely colored on a black background — visually matching a provided reference image.

Two independent approaches were developed and compared:

1. **Classical CV Pipeline** — OpenCV + Watershed + Ellipse Fitting (no deep learning)
2. **SAM-Based Deep Learning Pipeline** — Meta's Segment Anything Model with CLAHE pre-processing and geometric filtering

---

## Sample Results

| Input | Expected Output |
|-------|-----------------|
| Raw scan with overlapping grains on dark background | Each grain as a distinctly colored region on a black background |

| Approach | Grains Detected | Processing Time | Environment |
|----------|----------------|-----------------|-------------|
| Classical (OpenCV) | ~139 | ~5 seconds | CPU-only |
| SAM v2 (Deep Learning) | 208 | ~162 seconds | Google Colab (Tesla T4 GPU) |

---

## Evaluation Criteria

Both pipelines were evaluated against the following criteria:

- **Visual quality** of generated output
- **Ability to separate and represent individual grains**
- **Practical implementation quality**
- **Engineering creativity and experimentation**
- **Code structure and reproducibility**

---

# Approach 1: Classical Computer Vision Pipeline

**Notebook:** `rice_segmentation.ipynb`

### Method: OpenCV + Watershed + Ellipse Fitting

The pipeline uses a hybrid classical CV approach combining edge detection for grain separation, watershed for instance labeling, and ellipse fitting for clean shape rendering.

```
InputImage.jpg
      |
      v
+-------------------------------+
|  Step 1: Bilateral Filter     |  Edge-preserving noise smoothing
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 2: Otsu Thresholding    |  Binary mask (grains=white, bg=black)
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 3: Canny Edge Splitting |  Subtract grain edges -> separate touching grains
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 4: Morphological Cleanup|  Open (denoise) + Close (fill holes) on both masks
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 5: Local Maxima Seeds   |  Distance Transform -> scipy peak finding
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 6: Watershed            |  Seeds + full mask + original image gradients
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 7: Ellipse Fitting      |  cv2.fitEllipse per grain + size/shape filter
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 8: Colorization         |  Golden-ratio HSV hue stepping per grain
+---------------+---------------+
                |
                v
      segmented_output.png
```

### Step-by-Step Explanation

#### Step 1 — Bilateral Filter (Preprocessing)
- **Bilateral filter** smooths noise while preserving edges (unlike Gaussian blur which blurs everything uniformly)
- Critical for Step 3: grain boundaries must remain sharp for Canny edge detection
- Parameters: `d=9`, `sigmaColor=75`, `sigmaSpace=75`

#### Step 2 — Otsu's Thresholding
- Otsu automatically finds the optimal global threshold (~85) by minimizing intra-class variance
- Grains are bright (~130 avg intensity), background is dark (~26 avg)
- Produces ~66% foreground coverage — correct since grains cover most of the scanned area

#### Step 3 — Canny Edge Splitting (Key Innovation)
- **Problem:** At Otsu threshold, touching grains form one giant connected blob
- **Solution:** Canny edge detection finds the boundaries between adjacent grains
- Edges are dilated (thickened) then subtracted from the binary mask
- Result: touching grains become disconnected fragments — each with its own center

#### Step 4 — Morphological Cleanup (Dual Masks)
- **Split mask** (edge-subtracted): cleaned for seed detection — individual grains separated
- **Full mask** (original Otsu): cleaned for watershed foreground — full grain coverage
- Opening removes noise; closing fills holes

#### Step 5 — Distance Transform + Local Maxima Seeds
- Distance transform assigns each pixel its distance to the nearest background
- **scipy's `maximum_filter`** finds local maxima in a 40px neighborhood — each peak is a grain center
- More robust than a global threshold: finds peaks even in densely packed regions
- Produces ~190 seed markers (some will be merged by watershed)

#### Step 6 — Watershed Segmentation
- Seeds come from Step 5 (found on edge-split mask for better individual peaks)
- Foreground region comes from the **full** (unsplit) mask — bigger grain coverage
- Watershed runs on the **original color image** (richer gradient info than binary)
- Produces labeled regions per grain

#### Step 7 — Ellipse Fitting + Filtering (Key Innovation)
- **This is what makes the output look like the reference.** Instead of using raw jagged watershed masks, we fit smooth mathematical ellipses (`cv2.fitEllipse`) to each grain contour
- Rice grains are naturally ellipsoidal — fitted ellipses match reality
- Filtering removes noise (major < 15px), merged clusters (major > 150px), too-circular blobs, and thin slivers

#### Step 8 — Colorization
- **Golden-ratio hue stepping** (`phi=0.618`): consecutive hues are maximally separated
- HSV space with randomized saturation/value for visual variety
- Largest grains drawn first (z-order) so smaller grains render on top

### Classical Pipeline — Tunable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BILATERAL_D` | 9 | Larger = stronger smoothing (may blur small grains) |
| `CANNY_LOW/HIGH` | 30/90 | Lower thresholds = more edges detected (more splitting) |
| `EDGE_DILATE_ITER` | 2 | More dilation = thicker edge gaps (more separation) |
| `LOCAL_MAX_SIZE` | 40 | Larger = fewer seeds (less over-segmentation) |
| `MIN_PEAK_DISTANCE` | 6 | Higher = only strong peaks survive (fewer seeds) |
| `MAX_GRAIN_MAJOR` | 150px | Reject ellipses with major axis above this |
| `MIN_GRAIN_ASPECT` | 1.05 | Reject nearly-circular detections |

### Classical Pipeline — Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Grains form one giant connected blob at Otsu threshold | Canny edge subtraction splits them before seed detection |
| Raw watershed masks have jagged, irregular shapes | Ellipse fitting smooths each grain into a natural shape |
| Over-segmentation (too many seeds) | Local maxima with 40px neighborhood + ellipse area filtering |
| Under-segmentation (merged clusters) | Edge-split mask for seed finding + full mask for watershed foreground |
| Gaussian blur destroys grain edges | Bilateral filter preserves edges while smoothing noise |

---

# Approach 2: SAM-Based Deep Learning Pipeline

**Notebook:** `rice_grain_segmentation_SAM_v2_(2) (1).ipynb`

### What is SAM?

**Segment Anything Model (SAM)** is a foundation model for image segmentation introduced by Meta AI Research (Kirillov et al., 2023). It was trained on the **SA-1B dataset** — over 11 million images and 1.1 billion masks — giving it **zero-shot generalisation** capability: it can segment objects it has never seen before, including rice grains.

### SAM Architecture

SAM consists of three core components:

1. **Image Encoder (ViT Backbone):** A pre-trained Vision Transformer (ViT) processes the input image *once* to produce a dense feature embedding. Three model sizes:
   - **ViT-H** (Huge): 632M parameters, ~2.4 GB checkpoint — highest accuracy
   - **ViT-L** (Large): 308M parameters, ~1.2 GB checkpoint — strong accuracy, lower resource needs
   - **ViT-B** (Base): 91M parameters, ~375 MB checkpoint — lightweight fallback

2. **Prompt Encoder:** Encodes user-supplied prompts (points, bounding boxes, or coarse masks) into embedding vectors. In our pipeline, `SamAutomaticMaskGenerator` automates this by generating a **dense grid of point prompts** across the image.

3. **Mask Decoder:** A lightweight transformer-based decoder takes the image embedding and prompt embeddings to predict segmentation masks with confidence scores.

### SAM v2 Pipeline Flow

```
InputImage.jpg
      |
      v
+-------------------------------+
|  Step 1: Image Upload         |  Load & inspect (1024×847 px)
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 2: CLAHE Enhancement    |  Contrast-boost in LAB color space
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 3: VRAM-Aware Model     |  Auto-select ViT-H / ViT-L / ViT-B
|         Selection             |  based on available GPU memory
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 4: SamAutomatic         |  Dense point grid -> 295 raw masks
|         MaskGenerator         |
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 5: Geometric Filtering  |  Area + aspect ratio + solidity
|                               |  295 -> 208 masks
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 6: 3-Stage Smoothing    |  Morph close -> Gauss blur -> re-threshold
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 7: Colorization         |  Evenly-spaced HSV hues, area-sorted
+---------------+---------------+
                |
                v
+-------------------------------+
|  Step 8: Visualization        |  Side-by-side comparison
+---------------+---------------+
                |
                v
      sam_output.png
```

### Step-by-Step Explanation

#### Step 1 — Image Upload & Inspection
The raw scanned image (`InputImage.jpg`, 1024×847 px) is loaded and its dimensions verified.

#### Step 2 — CLAHE Pre-processing (Key Innovation in v2)
**Contrast Limited Adaptive Histogram Equalisation (CLAHE)** is applied to the luminance (L) channel in LAB colour space *before* feeding the image to SAM.

**Why CLAHE?** The raw scanned image has uneven illumination — some regions have low grain-background contrast. Standard SAM struggles to detect boundaries there. CLAHE fixes this by:
- Dividing the image into small tiles (8×8 grid)
- Performing histogram equalisation *locally* within each tile
- Applying a clip limit (`clipLimit=3.0`) to prevent noise amplification
- Interpolating tile boundaries for seamless output

Result: grain boundaries are uniformly sharp across the entire image.

#### Step 3 — VRAM-Optimised Model Selection (Engineering Innovation)
The pipeline implements **dynamic model selection** based on available GPU memory:

```python
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 16:
        model_type = "vit_h"    # Best accuracy
    else:
        model_type = "vit_l"    # Good balance
else:
    model_type = "vit_b"        # CPU fallback
```

This ensures optimal execution on any hardware — from A100 GPUs (ViT-H) to Colab T4s (ViT-L, ~15.6 GB VRAM) to CPU-only environments (ViT-B).

#### Step 4 — Automatic Mask Generation
`SamAutomaticMaskGenerator` is the core of the pipeline:
1. Generates a **dense grid of point prompts** across the entire image
2. Runs each prompt through SAM's mask decoder to produce candidate masks
3. Applies non-maximum suppression (NMS) to remove duplicates
4. Returns all candidates with metadata (area, stability score, bounding box)

**Result:** 295 raw candidate masks generated in ~162 seconds on T4 GPU with ViT-L.
- Area range: 14 – 53,925 px²
- Median area: 3,562 px²

#### Step 5 — Geometric Shape Filtering (Key Innovation in v2)
Raw SAM masks include non-grain detections (background blobs, dust, reflections). Multi-criteria geometric filtering removes them based on rice grain morphology (elongated ellipses):

| Filter | Threshold | What it Removes |
|--------|-----------|-----------------|
| `MIN_AREA` | 300 px² | Dust, noise, tiny artefacts |
| `MAX_AREA_FRACTION` | 4% of image | Large background blobs (tightened from 15% in v1) |
| `MIN_ASPECT_RATIO` | 1.3 | Nearly-circular masks (reflections, corners) |
| `MAX_ASPECT_RATIO` | 9.0 | Impossibly thin slivers |
| `MIN_SOLIDITY` | 0.72 | Highly irregular, non-convex blobs |

**Implementation:** For each mask, `cv2.findContours` extracts the largest contour, `cv2.fitEllipse` computes the aspect ratio, and solidity = (contour area) / (convex hull area).

**Filtering results:** 295 raw → **208 accepted** (87 rejected: 47 too small, 1 too large, 21 bad aspect ratio, 18 low solidity).

#### Step 6 — 3-Stage Mask Smoothing (Key Innovation in v2)
Raw SAM masks have jagged, staircase-like boundaries. A three-stage smoothing pipeline fixes this:

1. **Morphological closing** (`MORPH_CLOSE`, elliptical kernel k=7, 2 iterations): fills holes and gaps
2. **Gaussian blur** (kernel 11×11, auto-sigma): creates soft alpha edges
3. **Re-threshold at 0.5**: converts back to clean, smooth binary mask

Result: soft, rounded grain outlines matching reference output quality.

#### Step 7 — Colorization
Filtered and smoothed masks are painted onto a black canvas:
- Evenly-spaced HSV hues with randomised saturation [0.75, 1.0] and value [0.80, 1.0]
- Masks **sorted by area (largest first)** — smaller grains render on top for visibility
- Result: **208 distinctly coloured grain instances**

#### Step 8 — Visualization
Side-by-side comparison of original input vs. colorized segmentation output.

### SAM v1 → v2 Evolution

| Issue in v1 | Fix in v2 |
|-------------|-----------|
| Large background blob incorrectly coloured | Tighter `MAX_AREA_FRACTION` (15% → 4%) + geometric filters |
| Jagged / harsh grain boundaries | 3-stage smoothing: morph close → Gaussian blur → re-threshold |
| Used ViT-B (lighter model) | VRAM-aware auto-selection: ViT-H → ViT-L → ViT-B fallback |
| No foreground–background pre-separation | CLAHE pre-processing boosts contrast uniformly |
| Simple area-only filtering | Multi-criteria: area + aspect ratio + solidity |

---

# Comparative Analysis

| Criterion | Classical (OpenCV) | SAM v2 (Deep Learning) |
|-----------|--------------------|------------------------|
| **Visual quality** | Good — smooth ellipses but geometric approximation | Excellent — true grain contours with smooth boundaries |
| **Instance separation** | ~139 grains; struggles with dense clusters | 208 grains; object-aware masks handle overlaps |
| **Boundary quality** | Smooth but approximate (fitted ellipses) | Natural contours after mask post-processing |
| **Computational cost** | Fast — CPU-only, ~5 seconds | GPU required; ~162 seconds on T4 |
| **Robustness** | Sensitive to hyperparameters and lighting | Generalises well across conditions |
| **Dependencies** | OpenCV, NumPy, SciPy (lightweight) | PyTorch, SAM checkpoint (1.2–2.4 GB) |
| **Engineering creativity** | Canny edge subtraction + ellipse fitting | CLAHE + geometric filtering + 3-stage smoothing |
| **Code structure** | Modular 8-step pipeline; each step visualised | Modular 8-step pipeline; well-documented functions |

### Key Findings

1. **SAM provides superior instance separation**, particularly in dense-cluster regions where classical watershed with edge subtraction still produces under- or over-segmented results.
2. **The classical pipeline's ellipse fitting** produces aesthetically pleasing outputs that approximate rice grain shapes effectively, despite the loss of true grain geometry.
3. **For resource-constrained environments**, the classical pipeline remains a practical and fast solution.
4. **CLAHE pre-processing** significantly improved SAM's boundary detection, particularly in low-contrast image regions.

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip
- For SAM pipeline: Google Colab or GPU environment with ≥8 GB VRAM

### Installation (Classical Pipeline — Local)

```bash
# Navigate to project directory
cd Rice-Grain-Segmentation

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Run the Classical Pipeline

**Option A: Jupyter Notebook**
```bash
jupyter notebook rice_segmentation.ipynb
# Run all cells (Kernel -> Restart & Run All)
```

### Run the SAM Pipeline

**Google Colab (recommended):**
1. Upload `rice_grain_segmentation_SAM_v2_(2) (1).ipynb` to Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → T4 GPU`
3. Upload `InputImage.jpg` when prompted
4. Run all cells — the notebook auto-downloads SAM checkpoints

---

## Project Structure

```
Rice-Grain-Segmentation/
|-- InputImage.jpg                                      # Raw scanned rice grain image (input)
|-- ExpectedOutput.jpeg                                 # Reference segmented image (target)
|-- rice_segmentation.ipynb                             # Classical pipeline (OpenCV + Watershed)
|-- rice_grain_segmentation_SAM_v2_(2) (1).ipynb        # SAM v2 pipeline (Deep Learning)
|-- requirements.txt                                    # Python dependencies (classical pipeline)
|-- README.md                                           # This file
|-- LICENSE                                             # License file
+-- output/
    |-- segmented_output.png                            # Classical output (~139 grains)
    |-- diagnostic.png                                  # 6-panel pipeline diagnostic
    |-- step1_preprocess.png ... step8_comparison.png   # Per-step visualisations
    +-- full_diagnostic.png                             # 8-panel full diagnostic
```

---

## Dependencies

### Classical Pipeline
| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | >= 4.8.0 | Core CV: bilateral filter, Otsu, Canny, watershed, ellipse fitting |
| `numpy` | >= 1.24.0 | Array operations |
| `matplotlib` | >= 3.7.0 | Visualization and figure saving |
| `scipy` | >= 1.11.0 | `maximum_filter` for local maxima seed detection |
| `Pillow` | >= 10.0.0 | Image I/O |

### SAM Pipeline (installed automatically in Colab)
| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0 | Deep learning backend |
| `segment-anything` | latest | SAM model and mask generator |
| `opencv-python` | >= 4.8.0 | CLAHE, contour analysis, morphological operations |
| `scipy` | >= 1.11.0 | Mask labeling utilities |

---

## Future Improvements

1. **MobileSAM / FastSAM**: Deploy a lightweight SAM variant for real-time or low-resource inference
2. **Fine-tuning SAM**: Train SAM on a dataset of grain images for domain-specific accuracy
3. **Adaptive local maxima**: Use grain-size-adaptive neighborhood instead of fixed 40px (classical)
4. **Hybrid approach**: Use SAM for initial mask proposals and classical ellipse fitting for shape regularisation
5. **Colour palette matching**: Sample colours directly from the reference output for exact visual fidelity
6. **Multi-scale watershed**: Run the classical pipeline at multiple distance threshold levels and merge results

---

