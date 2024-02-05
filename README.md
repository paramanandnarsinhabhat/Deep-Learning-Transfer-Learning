
# DEEP-LEARNING-TRANSFER-LEARNING

This project is focused on applying transfer learning techniques using the VGG16 model to classify vehicles into emergency and non-emergency categories.

## Getting Started

To get started with this project, clone the repository and install the required dependencies listed in the `requirements.txt`.

### Prerequisites

You need to have the following packages installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- scikit-image
- tensorflow (includes Keras)

Install all the required packages by running:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project should consist of vehicle images labeled as emergency or non-emergency vehicles. The labels should be contained in a CSV file named `emergency_classification.csv`, which needs to be placed in the `data/` directory. Images should be placed in the `data/images/` directory.

## Usage

Run the Python scripts located in the `source/` directory or execute the Jupyter notebook in the `notebook/` directory to perform the classification.

## Structure

The project has the following structure:

- `data/`: Contains the dataset and images.
- `notebook/`: Contains Jupyter notebooks for experimentation.
- `source/`: Contains the source code in Python script format.
- `requirements.txt`: Lists all the dependencies for the project.

## Model Training and Evaluation

The project uses the VGG16 model pre-trained on ImageNet. The steps involved are:

1. Load and preprocess the data.
2. Split the dataset into training and validation sets.
3. Utilize the VGG16 model to extract features.
4. Fine-tune a neural network model on top of these features.
5. Compile and train the model.
6. Evaluate the model's performance on the validation set.

The code provided in the `source/` directory guides through these steps, culminating in a printout of the validation accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- The VGG16 model, pre-trained on ImageNet.
- The creators of the utilized Python packages.
```

