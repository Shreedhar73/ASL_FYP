# ASL_FYP is a modern application that will detect the sign of disabled person who cannot talk and  our application will convert those sign into text as well as speech.

#We have provided a web Based Interface using Python's Flask library , so that you could train the model yourself without having to code whole CNN Your self. Plus you can also view the distribution of Alphabets on Training DataSet as a Histogram.

After Training you can obtain Model file in project root directory and also have option to convert the Saved Model to .tflite , so that it can be used in Flutter Application.



This repo  contains Two Flutter Application :
    ASL_FlutterA_image_picker :: Detects Picked Image (user Flutter Version 2.8.1)
    ASL_Flutter_Live_Detection :: Detects Image from Live Video Stream(Less Accurate)(Flutter Version : 2.10.0)

Here, we have a mobile application that will open camera  and capture the sign and detect that sign based on model provided using python . We have build a web app that hepls to generate different model based on given input value.

