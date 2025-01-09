# Binary-Classification-with-Transfer-Learning
Binary Classification with Transfer Learning Using ResNet50 : Car Detection 

**1. Background and Significance**

**Research Problem**  
This project investigates vehicle detection in images using a binary classification approach enhanced by transfer learning. The objective is to determine whether a given image contains a car or not, leveraging the ResNet-50 architecture—a cutting-edge convolutional neural network model.

**Importance**  
Vehicle detection plays a pivotal role in domains such as autonomous driving, traffic control, and security systems. Accurate and efficient detection can enhance road safety, streamline traffic flow, and strengthen automated surveillance mechanisms.

**Potential Impact**  
The project demonstrates the practical application of transfer learning with ResNet-50 for binary classification tasks, emphasizing reduced training time and high accuracy on image datasets. It underscores the criticality of data preprocessing and model optimization in achieving reliable results for real-world applications.

**Gap Addressed**  
By employing transfer learning with pre-trained weights, this project addresses the challenges of applying deep learning to small datasets, enhancing generalization and improving model performance.

---

**2. Literature Review**

**Relevant Work**  
- Effectiveness of ResNet-50 in image classification tasks.  
- Transfer learning strategies to enhance model performance on limited datasets.  
- Data augmentation methods to expand dataset diversity and variability artificially.  

**Current State**  
Although ResNet-50 has shown significant success in various image classification problems, its application in binary classification on small datasets presents challenges, particularly in minimizing overfitting and ensuring robust performance.

---

**3. Data and Methods**

**Data Sources**  
The dataset includes images classified into two categories: 'Car' and 'Not Car.' It was sourced from publicly available repositories on Kaggle, including:  
- [Cars Image Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset)  
- [Universal Image Embeddings Dataset](https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings)

**Methodology**  
- **Model Architecture**: The ResNet-50 model was fine-tuned for binary classification by modifying its output layer to a single neuron with a sigmoid activation function.  
- **Data Augmentation**: Techniques such as rotation, flipping, and scaling were implemented to increase dataset diversity.  
- **Training Strategy**: The model was trained using the Adam optimizer, with a learning rate scheduler, early stopping, and model checkpointing to optimize the process.  

**Evaluation Metrics**  
Performance was assessed using:  
- Accuracy  
- Precision, Recall, and F1-Score  
- Confusion Matrix  

---

**4. Results and Analysis**

**Training Performance**  
The model achieved 95% accuracy on the training set and 93% on the validation set after 25 epochs, with a consistent loss convergence indicating effective training.

**Confusion Matrix**  
Analysis of the confusion matrix showed high precision and recall for both classes, with only minor misclassifications.  

**Visualization**  
Graphs of training and validation accuracy/loss trends illustrated steady improvements during training, with minimal overfitting observed.

---

**5. Learning Outcomes**

**Technical Skills**  
- Gained expertise in transfer learning with ResNet-50.  
- Enhanced proficiency in data augmentation and hyperparameter optimization.  

**Challenges**  
- Addressing dataset imbalances and ensuring generalization.  
- Successfully fine-tuning ResNet-50 for a binary classification problem.  

**Insights**  
Effective data preprocessing and augmentation were crucial to achieving high model performance. Adjusting parameters like batch size and learning rates significantly influenced training efficiency.  

---

## Contact
If you have any questions or need further assistance, please feel free to contact:

- **Name**: Sarowar Alam
- **Email**: sarowaralam40@gmail.com
- **GitHub**: [https://github.com/SarowarAlam](https://github.com/SarowarAlam)

**Personal Growth**  
This project deepened my understanding of deep learning workflows and transfer learning techniques, equipping me with valuable skills for future endeavors.

**Note: Due to large dataset we uploaded 100 images per folder in dataset to set an example**
