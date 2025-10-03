
<div align="center">
	<h1>PosePilot 🧘</h1>
	<p>
		<a href="https://link.springer.com/chapter/10.1007/978-3-031-99568-2_17">
			<img src="https://img.shields.io/badge/Springer%20DOI-10.1007%2F978--3--031--99568--2_17-blue" alt="Springer DOI">
		</a>
		<a href="https://arxiv.org/abs/2505.19186">
			<img src="https://img.shields.io/badge/arXiv-2505.19186-b31b1b.svg" alt="arXiv">
		</a>
	</p>
    <p><b>A Novel Posture Correction System Leveraging BiLSTM and Multihead Attention</b></p>
	<img src="./assets/PosePilot-Pipeline.png" alt="PosePilot Pipeline" />
</div>

### Overview  📖

PosePilot is an edge-AI solution for real-time posture correction in physical exercises, with a focus on Yoga. It integrates pose recognition and personalized corrective feedback, leveraging BiLSTM and Multihead Attention for robust, lightweight, and accurate posture analysis. The system is designed for deployment on edge devices and can be extended to various at-home and outdoor exercises.

Key features:
- Automatic human posture recognition
- Personalized, instant corrective feedback at every stage
- Lightweight and robust model for edge deployment


<div align="center">
	<img src="./assets/False1.png" alt="PosePilot in Wild"/>
</div>


### Dataset 🗂️

Our in-house dataset was created with videos from four angles, featuring 14 participants (ages 17-25) performing six poses. Frame-by-frame keypoint extraction using Mediapipe identified 33 keypoints, with 17 used to compute 680 angles for pose analysis. The dataset contains 336 videos, filmed indoors with controlled lighting.

<div align="center">
	<img src="./assets/Dataset.png" alt="Dataset" width="60%"/>
</div>


## Citation  🏷️

If you use PosePilot in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-99568-2_17,
	author    = {Gadhvi, Rushiraj and Desai, Priyansh and Siddharth},
	title     = {PosePilot: An Edge-AI Solution for Posture Correction in Physical Exercises},
	booktitle = {Pattern Recognition and Image Analysis},
	year      = {2026},
	publisher = {Springer Nature Switzerland},
	address   = {Cham},
	pages     = {208--219},
	isbn      = {978-3-031-99568-2}
}
```
