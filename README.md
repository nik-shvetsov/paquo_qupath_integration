# How to run QuPath with distilled cell segmentation and classificaton model via paquo

## Setup environment

### Running with Docker
```bash
xhost +local:$USER
docker run --rm --gpus all -e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /path/to/slides:/workspace/data \
-v /path/to/qupath_projects:/workspace/projects \
nikshv/qupath:v0.5.1-mod-v2
```
### Manual approach

1. Download .p2 model from google drive
Link: https://drive.google.com/file/d/1R3_XETgTJtaco3ZLio9kYCFjOfHth8RF/view?usp=sharing

2. Install QuPath
See installation instructions on website (https://qupath.github.io)

3. Install conda
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh && rm Miniconda3-latest-Linux-x86_64.sh
```

4. Install libraries
```bash
conda create -n paquo -y python=3.12 pip openslide pyvips -c conda-forge
conda activate paquo
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install scipy polars numba shapely openslide-python scikit-image opencv-python paquo
```

5. Run QuPath from the conda environment
```bash
/path/to/QuPath
```
It is essential to run QuPath within created conda environment (i.e. paquo)


## Using the model

### Create project
1. Add slides to the project
2. Select regions to annotate on slides
All regions in the project will be tagged for annotation.

### Run script from QuPath
1. Press (Automate -> Script Editor) in main QuPath window.
2. Paste 'script.groovy' content to the window
3. Edit path variables of the script, model and the project in groovy script. All of them should be full system paths.
4. Press 'Run' button
5. If it successfully finished, close Script Editor and press (File -> Reload data) in main QuPath window.
6. Inspect results

### 
- Don't forget to save annotations before running script and reload data in QuPath to see produced detections.
- QuPath have autocropping feature enabled by default as well as the script. If you want to validate detections using external viewer or script, consider setting 'use_autocrop' to 'False' in script.groovy or in run.py if you use it directly.
