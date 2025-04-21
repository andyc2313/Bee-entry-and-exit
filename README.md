# Bee Tracking and Counting System
Bee Tracking Demo （https://youtu.be/IL1JaYVnD24）

---

# Bee Tracking and Counting System 🐝📊

## Overview

This repository contains a computer vision-based system for tracking and counting honeybees entering and exiting a beehive 🏠🐝. The system uses YOLOv8 for detection and a custom tracking algorithm to maintain consistent bee identities 🆔, enabling accurate counting of bee movements. 📈

### Key Features

- **Real-time bee detection** using a trained YOLOv8 model optimized for honeybees 🐝
- **Custom tracking algorithm** to prevent ID switching 🚫🔄 and maintain consistent identities
- **Entry/exit counting logic** based on movement trajectories and appearance changes 🔄➡️⬅️
- **Visualization tools** for displaying tracking results and counts 
- **Region of Interest (ROI)** based filtering to focus on the beehive entrance 

## Requirements 📋

- Python 3.8+
- OpenCV 4.5+
- Ultralytics YOLOv8 🐝💻
- FilterPy
- SciPy
- NumPy

## Installation 🛠️

1. Clone this repository:
   ```bash
   git clone https://github.com/username/bee-tracking.git
   cd bee-tracking
   ```

2. Create a conda environment (optional but recommended):
   ```bash
   conda create -n bee-tracking python=3.8
   conda activate bee-tracking
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained YOLOv8 model or train your own (see [Training](#training) section). 

## Usage 🎬

### Basic Usage

Run the bee tracker on a video file:

```bash
python main.py --video path/to/video.mp4 --model path/to/model.pt
```

### Command Line Arguments ⚙️

- `--video`: Path to input video (required)
- `--model`: Path to YOLOv8 model weights (default: './weights/best.pt') 🎯
- `--conf`: Detection confidence threshold (default: 0.3) 📏
- `--output`: Path to save output video (optional) 🎥
- `--log`: Path to save counting log (optional) 📊
- `--roi`: Region of interest coordinates as x1,y1,x2,y2 (default: "250,648,2250,870") 🗺️

### Example

```bash
python main.py --video data/hive_video.mp4 --model weights/bee_model.pt --conf 0.35 --output results/processed.mp4 --log results/counts.txt
```

## System Architecture 🏗️

### Components

1. **YOLOv8 Detector**: Identifies bees in each frame 
2. **IDSwitchPreventer**: Maintains consistent bee identities using spatial and appearance cues 
3. **BeeCounter**: Analyzes trajectories to count entrance and exit events 
4. **Visualization Module**: Creates visual feedback of tracking results 

### Workflow

1. Frame acquisition from video source 🎥
2. Bee detection using YOLOv8 🐝🔍
3. Filtering detections based on Region of Interest (ROI) 🗺️
4. Tracking ID assignment and maintenance using custom algorithm 🆔⚙️
5. Analysis of bee movements to count entries and exits ➡️⬅️
6. Visualization of results 📊🎨

## Algorithm Details 🔍

### ID Switch Prevention 🚫🔄

The system uses a custom algorithm to prevent ID switching, maintaining consistent tracking across frames:

1. **Spatial matching**: Uses distance and IoU metrics 📏
2. **Direction consistency**: Analyzes movement patterns ➡️⬅️
3. **Appearance modeling**: Considers size changes 🔲↔️
4. **Lost track recovery**: Re-identifies bees that temporarily disappear 🔄

### Entry/Exit Counting Logic 

Bees are counted based on:

1. **Trajectory analysis**: Direction of movement ➡️⬅️
2. **Position relative to ROI**: Entry to specified regions 🗺️
3. **Size changes**: Expansion/contraction of detected bounding boxes 📏
4. **Appearance duration**: Time spent in tracking ⏳
