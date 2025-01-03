# Sign Language Detection


This project is a real-time sign language detection system that uses MediaPipe Hands for hand landmark detection, a trained k-NN classifier for gesture recognition, and pyttsx3 for converting recognized gestures into speech. 
It captures video via webcam, processes frames to detect gestures, and overlays results on the video feed, which is streamed to a web-based interface using Flask. 
The system features a speech thread for smooth text-to-speech output, ensuring gestures are spoken aloud without disrupting performance. 
With its modular design, the application is ideal for assisting individuals with hearing impairments or serving as a foundation for advanced human-computer interaction systems.

## Features
- Real-time hand gesture detection and recognition.
- Supports American Sign Language (ASL).
- Converts recognized gestures into text and speech.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Sign-Language-Detection.git
   
2. In your terminal :
   ```bash
   cd Sign-Language-Detection
   
3. 
   ```bash
   pip install -r requirements.txt


4. 
   ```bash
   python app.py



