# 🚦 Lane Violation and Vehicle Recognition System

An AI-powered computer vision application that automatically detects lane violations and recognizes vehicle registration numbers from traffic footage. The system helps improve traffic monitoring by identifying vehicles that violate lane rules and generating automated alerts.

## 🚀 Features

* Detects lane violations in real-time
* Automatic vehicle number plate recognition (ANPR)
* Captures evidence of traffic violations
* Sends email notifications with violation details
* User-friendly web interface
* Built using Python and Computer Vision techniques

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Framework:** Flask
* **Computer Vision:** OpenCV
* **Machine Learning / OCR:** EasyOCR / OCR-based text recognition
* **Frontend:** HTML, CSS, JavaScript
* **Database:** (Add if applicable)
* **Email Service:** Gmail SMTP
* **Environment Management:** Python Virtual Environment

## 📂 Project Workflow

1. Capture video stream or uploaded traffic footage.
2. Detect lane boundaries.
3. Identify vehicles crossing restricted lanes.
4. Extract the vehicle number plate.
5. Recognize the registration number using OCR.
6. Store the violation details.
7. Send an automated email notification with the captured evidence.

## 📦 Installation

### Clone the repository

```bash
git clone git@github.com:GI-AI-Projects/Lane_Violation_and_Recognition_System.git
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file and add:

```env
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_generated_app_password
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:5000
```

## 📸 Output

The application detects lane violations, recognizes the vehicle number plate, records the violation, and can send an email notification containing the violation details and captured evidence.

## 🎯 Future Improvements

* Multiple lane support
* Live CCTV integration
* Vehicle type classification
* Dashboard for traffic analytics
* Cloud deployment
* Fine management system
* Real-time monitoring

## 👨‍💻 Author

**Akshay Sharma**

* GitHub: https://github.com/akshay79068
* LinkedIn: https://linkedin.com/in/akshay-sharma-63a704259

If you found this project useful, consider giving it a ⭐ on GitHub.
