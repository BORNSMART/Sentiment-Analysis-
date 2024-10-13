# EmotiSense

**EmotiSense** is a mental health prediction tool that uses natural language processing to analyze user input and predict mental health conditions with **94% accuracy**. The model is powered by a BERT classifier, utilizing transformers and torch, to deliver high-precision predictions based on text data.

This project combines a Flask-based backend for handling requests and processing predictions with a simple and responsive HTML/CSS frontend for an intuitive user experience.

## Features
- **94% Accuracy**: Achieved using a BERT classifier.
- **Real-time Predictions**: Users can input text directly via the frontend, and receive immediate predictions.
- **User-Friendly Interface**: The frontend is designed using HTML and CSS for ease of use and accessibility.
- **Efficient Backend**: Flask is used to ensure a smooth, lightweight backend for handling requests.

## Technologies Used
- **Model**: BERT classifier, transformers, and torch.
- **Backend**: Flask.
- **Frontend**: HTML, CSS.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/BORNSMART/EmotiSense.git
   ```
2. Navigate to the project directory:
   ```bash
   cd EmotiSense
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/` to interact with the application.

## Usage
1. Enter text in the provided input box.
2. The system will analyze the text and predict the user's mental health state.
3. View the prediction result displayed on the same page.

## Model Training
The mental health prediction model is built using:
- **BERT Classifier** for natural language understanding.
- **Transformers** for text preprocessing and embeddings.
- **Torch** for model training and inference.

## Contributing
Feel free to submit pull requests or issues to enhance the functionality or accuracy of EmotiSense.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Project Description
**EmotiSense** is an AI-based mental health predictor tool designed to assist users in assessing their mental well-being based on text input. Utilizing advanced deep learning techniques with a BERT classifier, EmotiSense provides quick and reliable predictions with **94% accuracy**, offering a step forward in accessible mental health tools. With its easy-to-use interface and high accuracy, EmotiSense aims to provide meaningful insights to its users.
