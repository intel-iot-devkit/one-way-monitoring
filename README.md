# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
# Retail Pandemic Reference Implementation - One Way Monitor

| Details               |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 18.04 LTS   |
| Programming Language: |  Python* 3.5 |
| Time to Complete:     |  1 hour     |

![oneway](./doc/images/oneway.png)

## What it does

This reference implementation showcases a retail application which monitors if people is walking in the correct direction in a one-way direction store aisle and report the people walking in the wrong direction.

## Requirements

### Hardware

* 6th gen or greater Intel® Core™ processors or Intel® Xeon® processor, with 8Gb of RAM

### Software

* [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)

* [Intel® Distribution of OpenVINO™ toolkit 2020.3 Release](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

## How It works

The application uses the Inference Engine and Model Downloader included in the Intel® Distribution of OpenVINO Toolkit doing the following steps.

1. Ingests video from a file, processing it frame by frame.
2. Detects people in the frame of interest using a pre-trained DNN model.
3. Extract features from detected people to track them by using a second pre-trained DNN model.
4. Checks if any of the tracked persons is not going in the predefined and allowed direction.

The DNN models are optimized for Intel® Architecture and are included with Intel® Distribution of OpenVINO™ toolkit.

![architecture image](./doc/images/oneway-flow.png)

## Setup

### Get the code

Clone the reference implementation:

```bash
sudo apt-get update && sudo apt-get install git
git clone github.com:intel-iot-devkit/one-way-monitoring.git
```

### Install Intel® Distribution of OpenVINO™ Toolkit

Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

### Installing the requirements

To install the dependencies of the Reference Implementation, run the following commands:

```bash
  cd <path_to_oneway-monitoring-directory>
  pip3 install -r requirements.txt
```

### Which model to use

This application uses the [person-detection-retail-0013](https://docs.openvinotoolkit.org/2020.3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) and [person-reidentification-retail-0300](https://docs.openvinotoolkit.org/2020.3/_models_intel_person_reidentification_retail_0300_description_person_reidentification_retail_0300.html) Intel® pre-trained models, that can be downloaded using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that is used by the application.

To install the dependencies of the RI and to download the models Intel® model, run the following command:

```bash
mkdir models
cd models
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP32
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-reidentification-retail-0300 --precisions FP32
```

The models will be downloaded inside the following directories:

```bash
- models/intel/person-detection-retail-0013/FP32/
- models/intel/person-reidentification-retail-0300/FP32/
```

### The Config File

The _config.json_ contains the path to the videos and models that will be used by the application and also the coordinates of the virtual line of the queue.

The _config.json_ file is of the form name/value pair, `video: <path/to/video/myvideo.mp4>`

Example of the _config.json_ file:

```bash
{
  "video": "path/to/video/myvideo.mp4",
  "pedestrian_model_weights": "models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.bin",
  "pedestrian_model_description": "models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml",
  "reidentification_model_weights": "models/intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.bin",
  "reidentification_model_description": "models/intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.xml",
  "coords": [[68, 64], [17, 67]],
  "area": 50
}
```

Note that __coords__ represents a virtual gate defined by two points with x,y coordinates and __area__ is the with in pixesl of the queue line area. 

### Which Input video to use

The application works with any input video format supported by [OpenCV](https://opencv.org/).

Sample video: https://www.pexels.com/video/a-crowd-of-shoppers-in-an-open-space-mall-3318080/

Data set subject to license https://www.pexels.com/license. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.

To use any other video, specify the path in config.json file.

## Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

__Note__: This command needs to be executed only once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again.

## Run the application

Change the current directory to the project location on your system:

```bash
cd <path-to-oneway-monitoring-directory>
```

Run the python script.

```bash
python3 oneway.py
```
