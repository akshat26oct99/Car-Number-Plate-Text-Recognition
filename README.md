# Automatic License Plate Detection

## Introduction

This project focuses on extracting the license plate number from a vehicle's image using advanced object detection and character recognition techniques. Automatic license plate detection technology enables the identification of vehicles by capturing and recognizing number plates from images provided by video surveillance cameras. This technology has numerous practical applications, such as toll gate operations, vehicle tracking, and locating stolen vehicles through CCTV footage.

## Approach

To accurately predict the license plate number, the following steps are performed:

1. **License Plate Detection**: 
   - The license plate is detected within the image using object detection methods like contour finding and YOLO (You Only Look Once).
   
2. **Character Segmentation**: 
   - Once the license plate is extracted, individual characters are isolated using segmentation techniques like finding rectangular contours.
   
3. **Character Recognition**: 
   - Finally, the segmented characters are recognized using deep learning classifiers. In this project, we utilized Convolutional Neural Networks (CNNs), which are highly effective for image-based tasks.

## Datasets

The following datasets were used at different stages of the project:

- **License Plate Detection (YOLO)**: 
  - Approximately 4,000 annotated images of vehicles with license plates.
  - Dataset: [Mendeley License Plate Dataset](https://data.mendeley.com/datasets/nx9xbs4rgx/2)

- **Character Recognition**: 
  - Around 1,000 images of digits (0-9) and alphabets (A-Z).
  - Dataset: [Character Dataset](https://link_to_character_dataset)

- **Testing the Complete Model**: 
  - About 200 images of vehicles with license plates.
  - Dataset: [Test Dataset](https://drive.google.com/file/d/1QAFdt5Mq8X6fZud7kdsjaJbJfSXrsFse/view?usp=sharing)

## Technologies & Tools

- **Python**: Core language used, version 3.6.
- **Jupyter Notebook**: IDE used for developing and testing the model.
- **OpenCV**: For image processing and computer vision tasks.
- **TensorFlow & Keras**: For building and training the deep learning models.
- **YOLOv3**: For real-time object detection.
- **Scikit-Learn**: For machine learning utilities.
- **Matplotlib**: For plotting and visualizing data.
- **Imutils**: For basic image processing functions using OpenCV.

## Methodology

### 1. License Plate Detection

- **Contour Method**:
  - Images are resized, converted to grayscale, and denoised.
  - Binarization is applied to simplify the detection.
  - Contours are detected and filtered based on their shape to identify the license plate.
  - Detected plates are straightened to facilitate further processing.

- **YOLOv3 Method**:
  - YOLOv3 was trained on a custom dataset to detect license plates.
  - The trained model outputs bounding boxes around detected plates.

### 2. Character Segmentation

- Pre-processed images are further refined to isolate individual characters.
- Contours of characters are extracted based on defined dimensions.

### 3. Character Recognition

- A CNN model is used to recognize characters from the segmented images.
- The model consists of multiple convolutional layers with ReLU activation, followed by max-pooling, dropout, and dense layers.
- Hyperparameters were optimized using Grid Search.

## Hyperparameter Tuning

Optimal parameters determined through tuning:

- **Dropout Rate**: 0.4
- **Learning Rate**: 0.0001
- **Optimizer**: Adam

## Results

- **Contour Method Accuracy**: ~60.24%
- **YOLOv3 Method Accuracy**: ~74.10%
- **Hybrid Approach Accuracy**: ~90.96%
  - The hybrid model first applies YOLOv3, and if no plate is detected, it falls back on the Contour method. This approach significantly improves accuracy.

## Optimizations

- The hybrid approach prioritizes YOLOv3 to minimize false positives, leading to more reliable detection.

## Conclusion

This project demonstrates the effective combination of object detection and character recognition techniques to build a robust automatic license plate detection system. The hybrid approach not only improves accuracy but also ensures reliable performance in real-world scenarios.
