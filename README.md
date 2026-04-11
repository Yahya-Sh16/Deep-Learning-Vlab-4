# Virtual Overfitting Lab

This repository contains the source code for the **Virtual Overfitting Lab**, an interactive educational web application designed to demonstrate the concept of overfitting using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset.

## Architecture

The project is split into two primary components:

- **Backend (`/backend`)**: A PyTorch-based API that manages the training processes for the CNN models. It supports configurable hyperparameters (such as learning rate, dropout, and weight decay) and streams real-time training metrics, including validation curves, back to the client.
- **Frontend (`/frontend`)**: A modern React/Next.js interface that guides students through the lab steps (Aim, Theory, Procedure, Simulation, etc.). It features dynamic dashboards for real-time training visualization, experiment comparisons, and interactive controls to induce and then mitigate overfitting.

## Directory Structure

- `backend/`: PyTorch model definitions, training loops, and API server logic.
- `frontend/`: Next.js frontend code, React components, state management, and styling.

## Getting Started

### Prerequisites

- Node.js and npm (for the frontend)
- Python 3.x and pip (for the backend)

### Running the Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install Python dependencies (ensure you have a virtual environment set up if preferred):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend server. *(Note: The exact command may vary depending on the specific API framework used, e.g., Flask, FastAPI, etc.)*
   ```bash
   python app.py
   ```

### Running the Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to `http://localhost:3000` to interact with the lab.

## Educational Objective

The primary objective of this virtual lab is to allow students to learn about model generalization through hands-on experimentation. By actively tuning configurations, users can visually witness the onset of overfitting and subsequently apply regularization techniques to see how model performance on unseen data improves.
