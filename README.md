# Evaluator 
The current demo for hand and bow detection is in src/computer_vision/hand_pose_detection, run test.py. Please change the path of the input videos and models on line 22, 87, and 93.

For help with installing any neccessary packages, check Evaluator_Vision_Team_Installation_and_Code_Setup_Documentation.pdf

# UI specific things
# For frontend: 
need npx and npm

navigate to the UI folder of the frontend directory
    if use terminal and are in the evaluator-code directory, do cd src/frontend/UI

to get it to start up, run npx expo start

Will use localhost 8081. on a web browser go to http://localhost:8081

All work for UI so far is in src/frontend/UI/app/index.tsx

You can also run an android emulator through expo. 

# For backend: 
python -m uvicorn routingAPI:app --reload

in the api folder to start

For mobile devices: 
To get things to work, use "python -m uvicorn routingAPI:app --host 0.0.0.0 --port 8000" instead of "python -m uvicorn routingAPI:app --reload" to launch the backend API. Change the value of server ip in line 39 in index.tsx and line 34 in routingAPI, and use the IP address of your laptop (where you're running the backend)
