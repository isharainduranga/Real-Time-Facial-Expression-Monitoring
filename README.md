
# Real-Time Facial Expression Monitoring

This project is a web-based application that monitors and detects real-time facial expressions. It is built using React with Tailwind CSS for the frontend and Flask for the backend. The model used for expression detection is trained using PyTorch and a Convolutional Neural Network (CNN) architecture based on ResNet18.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Contributing](#contributing)

## Features
- Real-time facial expression detection using webcam
- Classifies emotions into predefined categories (e.g., Angry, Disgust, Sad, Fear)
- User-friendly interface built with React and Tailwind CSS
- Backend API with Flask for serving the trained model
- Efficient performance with ResNet18 model optimized using PyTorch

## Technology Stack

### Frontend:
- **React**: JavaScript library for building user interfaces.
- **Tailwind CSS**: Utility-first CSS framework for custom designs.
  
### Backend:
- **Flask**: Python web framework for backend logic and API management.
  
### Machine Learning:
- **PyTorch**: Deep learning library used for training the CNN model.
- **ResNet18**: CNN architecture used for accurate and efficient facial expression recognition.

## Model Architecture
The facial expression detection model is built using a modified ResNet18 architecture. It is fine-tuned for recognizing four emotion categories: 
- Angry
- Disgust
- Sad
- Fear

The model takes grayscale, 48x48 pixel images as input and uses a Convolutional Neural Network (CNN) for feature extraction and classification.

## Setup and Installation

### Prerequisites:
- Node.js
- Python 3.x
- PyTorch
- Flask

### Setup Frontend:
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/your-username/real-time-facial-expression-monitoring.git
   cd real-time-facial-expression-monitoring
   \`\`\`
2. Navigate to the frontend folder and install dependencies:
   \`\`\`bash
   npm install
   \`\`\`
3. Start the React app:
\`\`\`bash
npm start
\`\`\`

### Python Setup(Backend):
1. Create a virtual environment and activate it:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   \`\`\`
3. Start the Flask server:
   \`\`\`bash
   python app.py
   \`\`\`

## Usage
1. Start both the frontend and backend servers.
2. Open the React application in your browser (usually \`http://localhost:3000\`).
3. Allow camera permissions, and the application will start detecting your facial expressions in real-time.

## Training the Model
The model is trained using a custom dataset of facial expressions, with the following steps:

1. Preprocess the dataset by resizing images to 48x48 pixels and converting them to grayscale.
2. Use PyTorch's ResNet18 architecture and fine-tune it to classify emotions.
3. Train the model using the Adam optimizer and CrossEntropyLoss.
4. Save the trained model to be used by the Flask backend for real-time inference.

If you'd like to retrain the model, follow the steps in the `emotion detection.ipynb` script.

## Contributing
Feel free to fork this repository, submit issues, or make pull requests to improve the project.


