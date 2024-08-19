# Control System Neural Network

## Overview

This project demonstrates the use of a feedforward neural network (FFNN) to approximate a function in a control system context. The script includes training a neural network model, visualizing the loss history, comparing the model results with reference data, and generating 3D plots of function values.

## Features

- **Neural Network Setup**: Defines a basic feedforward neural network to approximate a function.
- **Training**: Implements a training loop with a custom loss function that includes boundary conditions (BC), initial conditions (IC), and residuals.
- **Visualization**: Plots loss history over epochs, compares model results with reference data, and generates 3D visualizations of the function and training results.

## Requirements

- Python 3.x
- Required Python packages:
  - matplotlib
  - numpy
  - torch

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/control-system-neural-network.git
   cd control-system-neural-network
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Model Training**: The script initializes and trains a feedforward neural network to approximate a given function. It sets up the model, optimizer, and loss function, and performs training for a specified number of epochs.

2. **Visualization**: After training, the script plots:
   - The loss history over epochs.
   - A comparison between the model's results and the reference data.
   - 3D visualizations of the function and training results.

To run the script, use the following command:

```bash
python Control_system.py
```

## License

This project is licensed under the MIT License.
