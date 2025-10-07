# 📊 Data Schema

This directory contains the training and testing data for PosePilot's pose classification and correction models.

## 🗂️ Dataset Structure

The dataset is organized by yoga poses, with each pose captured from four different camera angles to provide comprehensive coverage of the movements:

```
data/
├── chair/
│   ├── cam_0/
│   │   ├── P1.csv
│   │   ├── P2.csv
│   │   └── ...
│   ├── cam_1/
│   ├── cam_2/
│   └── cam_3/
├── tree/
├── warrior/
├── downward_dog/
├── cobra/
└── goddess/
```

## 📋 Data Format

Each CSV file contains pose landmark data extracted from video frames using MediaPipe. The data follows this structure:

- **Rows**: Individual video frames
- **Columns**: Pose landmarks (33 keypoints × 3 coordinates = 99 columns) + metadata
- **Coordinate System**: Normalized coordinates (0-1 range) relative to image dimensions

## 🎯 Pose Landmarks

PosePilot uses MediaPipe's 33-point pose model for human pose detection. Each landmark represents a specific body part:

<div align="center">
    <img src="../assets/keypoint_map.png" alt="Pose Landmarks Map" width="400">
</div>

### Landmark Index Reference

| Index | Landmark | Description |
|-------|----------|-------------|
| 0 | nose | Center of the nose |
| 1 | left eye (inner) | Inner corner of left eye |
| 2 | left eye | Center of left eye |
| 3 | left eye (outer) | Outer corner of left eye |
| 4 | right eye (inner) | Inner corner of right eye |
| 5 | right eye | Center of right eye |
| 6 | right eye (outer) | Outer corner of right eye |
| 7 | left ear | Left ear |
| 8 | right ear | Right ear |
| 9 | mouth (left) | Left corner of mouth |
| 10 | mouth (right) | Right corner of mouth |
| 11 | left shoulder | Left shoulder joint |
| 12 | right shoulder | Right shoulder joint |
| 13 | left elbow | Left elbow joint |
| 14 | right elbow | Right elbow joint |
| 15 | left wrist | Left wrist joint |
| 16 | right wrist | Right wrist joint |
| 17 | left pinky | Left pinky finger |
| 18 | right pinky | Right pinky finger |
| 19 | left index | Left index finger |
| 20 | right index | Right index finger |
| 21 | left thumb | Left thumb |
| 22 | right thumb | Right thumb |
| 23 | left hip | Left hip joint |
| 24 | right hip | Right hip joint |
| 25 | left knee | Left knee joint |
| 26 | right knee | Right knee joint |
| 27 | left ankle | Left ankle joint |
| 28 | right ankle | Right ankle joint |
| 29 | left heel | Left heel |
| 30 | right heel | Right heel |
| 31 | left foot index | Left foot index toe |
| 32 | right foot index | Right foot index toe |

## 🧘‍♀️ Supported Poses

The dataset includes the following yoga poses:

- **Chair Pose** (`chair/`) - Utkatasana
- **Tree Pose** (`tree/`) - Vrikshasana  
- **Warrior Pose** (`warrior/`) - Virabhadrasana
- **Downward Dog** (`downward_dog/`) - Adho Mukha Svanasana
- **Cobra Pose** (`cobra/`) - Bhujangasana
- **Goddess Pose** (`goddess/`) - Utkata Konasana

## ✅ Data Quality

This dataset contains yoga pose data captured from four camera angles (cam_0, cam_1, cam_2, cam_3) for comprehensive movement coverage and for better generalization in model training. Each pose includes multiple participant recordings with diverse body types and execution styles. Pose sequences capture complete movements from start to finish, enabling temporal analysis of joint trajectories throughout each asana. For more , see the [PosePilot research paper](https://arxiv.org/pdf/2505.19186).

For more information about PosePilot's architecture and usage, see the main [README](../readme.md).
