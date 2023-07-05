# COVID-19 Chest X-ray Image Classification using CNN, VGG16, ResNet50

This repository contains a Convolutional Neural Network (CNN) implementation for classifying COVID-19 from chest X-ray images. The models used for classification are VGG16 and ResNet50, two popular deep learning architectures known for their excellent performance in image classification tasks.

## [Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/download?datasetVersionNumber=2)
Click on the Dataset to download the dataset.

The dataset used in this project consists of chest X-ray images collected from individuals with and without COVID-19. The dataset is preprocessed and divided into training and testing sets to train and evaluate the models effectively. Each image is labeled as COVID-19 positive or negative to facilitate supervised learning.

## Models
1. **CNN**: A custom Convolutional Neural Network architecture is developed specifically for this project. It consists of multiple convolutional layers, pooling layers, and fully connected layers, followed by a softmax activation function to classify the images.

2. **VGG16**: The VGG16 model is a widely adopted deep learning architecture known for its simplicity and excellent performance. It consists of several convolutional and pooling layers, followed by fully connected layers and a softmax activation for classification.

3. **ResNet50**: The ResNet50 model is a state-of-the-art deep learning architecture known for its skip connections, which help alleviate the vanishing gradient problem. It comprises several residual blocks and ends with fully connected layers and a softmax activation.

## Usage
1. Clone the repository:
   ```
   https://github.com/Subratamanna07/CapstoneProject_Covid-19_Chest_X-ray_Image_Classification_CNN.git
   ```

2. Install the required dependencies (e.g., TensorFlow, Keras).

3. Prepare the dataset:
   - Ensure your chest X-ray images are labeled and properly organized.
   - Split the dataset into training and testing sets.

4. Train the models:
   - Run the training script for each model (CNN, VGG16, ResNet50) separately.
   - Adjust hyperparameters such as learning rate, batch size, and number of epochs as needed.

5. Evaluate the models:
   - Run the evaluation script to assess the performance of each model on the test set.
   - Generate classification reports and confusion matrices to analyze the results.

6. Experiment and fine-tune:
   - Modify the models' architecture or hyperparameters to improve performance.
   - Explore different data augmentation techniques or regularization methods.
   - Repeat the training and evaluation steps to assess the impact of changes.

## Results
Document and showcase the results achieved by each model, including accuracy, precision, recall, and F1-score. Compare the performance of CNN, VGG16, and ResNet50 on the COVID-19 chest X-ray classification task. Provide insights into their strengths, limitations, and potential areas for improvement.

## Further Improvements
To further improve the models and their application, consider the following:

- Experiment with other pre-trained models and architectures to explore their performance and suitability for the classification task.
- Augment the dataset with additional labeled examples or consider using transfer learning techniques to leverage larger external 
  datasets.
- Explore ensemble methods by combining the predictions of multiple models to enhance classification accuracy.
- Deploy the best-performing model as a web or mobile application for real-time COVID-19 detection
## Conclusion
In conclusion, this project demonstrates the development and evaluation of deep learning models for COVID-19 chest X-ray classification. The models achieve high accuracy and provide valuable insights for the detection and diagnosis of COVID-19 cases. By leveraging pre-trained architectures and fine-tuning them for the specific task, accurate classification results can be obtained. The project highlights the potential of deep learning in aiding healthcare professionals in identifying COVID-19 cases from chest X-ray images.

