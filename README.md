# Mosaic

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

Panoramic mosaic construction from overlapping images (TAVC assignment). The pipeline detects SIFT features,
matches descriptors, estimates homographies via DLT, filters outliers, and blends the warped images onto a
common canvas.

![Panorama of Escena5 (Picos de Europa, 7 images)](assets/panorama_scene5.jpg)

## Installation

The project uses [pixi](https://pixi.sh) (conda-based) to manage the environment.

1. Install pixi (if not already installed):

   ```sh
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Install the project dependencies (Python, OpenCV, NumPy, Matplotlib):

   ```sh
   pixi install
   ```

## Usage

Run the pipeline with pixi. By default it processes all three scenes (`Escena3`, `Escena4`, `Escena5`):

```sh
pixi run python mosaic.py
```

Process specific scenes:

```sh
pixi run python mosaic.py Escena5
```

Run the detector ablation study (SIFT, ORB, KAZE, BRISK):

```sh
pixi run python mosaic.py --ablation Escena5
```

Compare the outlier-filtering methods (residual, geometric, RANSAC):

```sh
pixi run python mosaic.py --filter-comparison Escena5
```

Output panoramas and diagnostic figures are written to the project directory.

## Results

Panoramas reconstructed for the three scenes:

| Scene | Images | Panorama |
| --- | --- | --- |
| Scene 3 (Ría del Nervión) | 2 | ![Panorama of Scene 3](assets/panorama_scene3.jpg) |
| Scene 4 (Bermeo) | 3 | ![Panorama of Scene 4](assets/panorama_scene4.jpg) |
| Scene 5 (Picos de Europa) | 7 | ![Panorama of Scene 5](assets/panorama_scene5.jpg) |

Feature correspondences after outlier filtering (cyan: inliers; red lines: points above the error threshold):

![Filtered correspondences for Scene 3](assets/correspondences_scene3.png)
