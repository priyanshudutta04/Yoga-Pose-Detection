# Yoga-Pose-Detection
AI based yoga pose detection using CNN and YOLO

## Sample Video
https://github.com/user-attachments/assets/f954c507-7c93-4cb2-b80a-f11585491cc7

## About

This project uses YOLOv8 for real-time object detection and a TensorFlow model for yoga pose classification. It captures live video, detects the presence of a person, extracts and analyzes their pose to provide accurate yoga pose identification. This system helps ensure correct yoga practice by providing immediate feedback on pose accuracy.

## Data

The datataset used is the roboflow's  `Yoga Pose Computer Vision Project`, which includes numerous images for each of the 107 yoga poses.


Dataset Source Link: [data url](https://universe.roboflow.com/new-workspace-mujgg/yoga-pose)

## Usage

1. After downloading the dataset, unzip it and place in the root directory.
   
2. Clone the repository
```
git clone https://github.com/priyanshudutta04/Cats-Vs-Dogs.git
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run the Python File
```
python yoga-detection.py
```
<br/>

*Note: If GPU is available install `cuda toolkit` and `cuDNN` for faster execution during Model-Training*

## Contributing

Contributions are welcome! If you have ideas for improving the model or adding new features, please feel free to fork the repository and submit a pull request.

## Support

If you like this project, do give it a ⭐and share it with your friends



