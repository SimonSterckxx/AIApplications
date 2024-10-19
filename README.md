# AIApplications  
#Create a virtual environment  
python -m venv venv  

#Activate the virtual environment  
venv\Scripts\activate  

#Install the required packages  
pip install -r requirements.txt  




## Case:

We are tasked with finding a way to reduce the average time to intervention in cases of fall-related emergencies, including strokes, by using camera-based AI technology to detect incidents and notify emergency services promptly.

## Our Project's Objective:

Our project focuses on implementing and evaluating existing AI models capable of detecting fall-related emergencies, including strokes. The models will analyze video footage in real-time, identifying incidents where a person has fallen, and then trigger alerts to emergency responders. By integrating these pre-trained models into a real-time system, we aim to enhance fall detection capabilities in various real-world environments.

## Running the AI
python pytorch/inference.py sound_event_detection --model_type="Cnn14_DecisionLevelMax" --checkpoint_path="Cnn14_DecisionLevelMax_mAP=0.385.pth" --audio_path="resources/R9_ZSCveAHg_7s.wav" --cuda


## Sources:
We got the idea of using pre-trained YOLOv8 models from this video, showcasing an example on how its used for fall detection:
https://www.youtube.com/watch?v=wrhfMF4uqj8&t=376s
