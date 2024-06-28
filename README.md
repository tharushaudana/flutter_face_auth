# Face Authentication Application

This Flutter project implements face authentication using the FaceNet512 model, storing face data (as Float32 arrays) and names in Firebase Firestore.

## Features

- **FaceNet512 Model**: Utilizes the FaceNet512 model to encode facial features.
- **Firebase Firestore Integration**: Stores face data (as a Float32 array of length 512) and corresponding names (as strings) in Firestore.
- **Face Data Loading**: Loads all face data from Firestore upon app launch.
- **Face Data Storage**: Allows users to capture their face and store the data in Firestore.
- **Face Prediction**: Predicts the user's identity by comparing the captured face data with the stored data using cosine similarity.

## Dependencies

- camera: ^0.10.5+5
- google_mlkit_face_detection: ^0.9.0
- image: ^3.0.2
- tflite_flutter: ^0.10.3
- cloud_firestore: ^4.13.2
- firebase_core: ^2.23.0

## Screenshots

### App Interface
![App Interface](screenshots/01.jpg)

### Predicted Page
![Predicted Page](screenshots/02.jpg)

### Data Storage in Firestore
![Data Storage in Firestore](screenshots/03.png)

## Purpose

This application aims to assist in mobile face authentication purposes, leveraging advanced face recognition technology and cloud storage.

---

I hope this project helps you in implementing mobile face authentication functionalities efficiently.
