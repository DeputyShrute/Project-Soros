# Project-Soros
 
 A dissertation project looking to use machine learning and Japanese candlestick data to predict trends in forex markets.

 Ryan Easter EAS16635772

 This project is made up of two section:

 - Object Detector for Candlestick Patterns
 - Price Prediciton Scripts

## Object Detector

### Requirements:
- OpenCV
- Ubuntu
- GPU support (optional)

The object detector can be found on GitHub by the user AlexeyAB at this link: https://github.com/AlexeyAB/darknet
Within the Readme of the project is a detailed set of configurations however, another copy can be found below.

This project requires an Ubuntu OS for easiest installation.

- Clone the darknet repo into the top level of the Project-Soros repo
- Run the make command in the darknet folder (List of configurable parameters: https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make)
- Download the weights from: https://github.com/DeputyShrute/Project-Soros/blob/main/darknet%20requirements/yolo-obj_last.weights
- Copy the downloaded weights file to the `backup` folder inside the `darknet` folder
- Copy the `yolo-obj.cfg` from the `darknet_requirements` folder to the cfg folder inside `darknet`
- Copy the `obj.data`, `obj.names` and `test.txt` from the `darknet_requirements` folder to the data folder inside` darknet`
- Copy the `test_img` folder into the `darknet` folder
- Run the following command in the root of the `darknet` folder: `./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_last.weights`
- When the prompt for an image display, use the file location of the images from the `test_img` folder.
- Depending if a GPU has been configured the output will either display in the screen or in the `predictions.png` in the root of the darknet folder.

This is basic setup for the object detector. If a GPU has been utilised in the `makerfile` the detections will run faster.

## Price Prediction

All of the required scripts are within the repo and before they can be run, the `requirements.txt` needs to be installed:
```python  
pip install -r requirements.txt
```
The software can then be started using by running:
``` python
python3 run.py
```
This is in the root of the Project-Soros Folder.
<<<<<<< HEAD

On the menu two options will appear:
- `Update all data` pulls fresh data from Yahoo
- `Full Analysis` runs both the object detection and price prediction
=======
>>>>>>> efe3477d0f801da32d14360f4aa6426ae148da02
