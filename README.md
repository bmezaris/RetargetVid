
# RetargetVid

## Introduction
This repository contains **RetargetVid**, a video retargeting dataset and SmartCrop, a fast cropping method for video retargeting.

We observed that each literature work about video retargeting uses an arbitrary selection of videos to test its results, and most of the time these videos are not provided, while the evaluation procedure relies on visual inspection of selected frames. Motivated by this we construct and release RetargetVid, a  publicly available benchmark dataset for video cropping, annotated by 6 human subjects.

We also release a new, rather simple, yet fast and well-performing, video cropping method, which selects the main focus out of the multiple possible salient regions of the video by introducing a new filtering-through-clustering processing step.

## Dataset
We selected a subset of 200 videos from the publicly available videos of the [DHF1k](https://github.com/wenguanwang/DHF1K) dataset. Specifically, we selected the first 100 videos of the training set (videos with indices 001 to 100) and the 100 videos of the validation set (videos with indices 601 to 700). All videos are in 16:9 aspect ratio and most of them consist of a single shot. The DHF1k dataset was originally constructed as a benchmark for evaluating visual saliency methods. Upon visual inspection, this dataset was considered ideal to provide a balanced set of both easy and challenging videos for the video cropping task. 

We invited 6 human subjects and asked them to select the region of each frame that would be ideal to be included in a cropped version of the video. Specifically, we assigned them the task of generating two cropped versions for each video, one with target aspect ratio of 1:3 and another one with target aspect ratio of 3:1. We selected these extreme target aspect ratios (despite not being used in real-life applications) in order to identify human preferences under very demanding circumstances. Moreover, less extreme target aspect ratios can still be evaluated by assessing to what extent an e.g. 9:16 crop window includes the 1:3 manually specified window.

Our crop window annotations for each video are in the form of text files, where the *i*-th line contains the top-left coordinates of the crop window for the *i*-th frame. There are 2400 annotation text files in total (200 videos * 2 target aspect ratios * 6 annotator subjects. The annotation text file are named *$video_id$-$target_aspect ratio$.txt*, where *$video_id$* is the original video filename and $target_aspect ratio$ is the target aspect ratio (e.g. 1-3 or 3-1). The annotation text files can be found in the *annotations* folder of this repository.

## Annotator Software
To assist the annotators in their task we implemented a graphical user interface tool which facilitates the navigation throughout the video, allows the user to set a crop window for each frame through simple drag-and-drop mouse operations, and overlays the crop window on the video frames to allow for the quick inspection of the user's decisions.

The software was implemented in C# and the source code is available in the *annotator_software* folder of this repository, in separate sub-folder for each subject.


## Method
We argue that cropping methods are more suitable for video aspect ratio transformation when the minimization of semantic distortions is a prerequisite. For our method, we utilize visual saliency to find the image regions of attention, and we employ a filtering-through-clustering technique to select the main region of focus. Therefore, we present a new, rather simple, yet fast and well-performing, video cropping method, which selects the main focus out of the multiple possible salient regions of the video by introducing a new filtering-through-clustering processing step.

Our method is implemented in Python 3 and the source is available in the *smartcrop* folder of this repository.

## Prerequisities
To run our SmartCrop method you will need Python 3. You must also wave installed the following packages (in parenthesis are the recommended packages version to instalL):

* TensorFlow (tensorflow-gpu==1.14.0)
* PyTorch (torch==1.7.1+cu110)
* ffmpeg (ffmpeg==1.4)
* OpenCV (opencv-python==4.2.0)
* SciPy (scipy==1.5.1)
* hdbscan (hdbscan==0.8.26)

To run the annotator software you will need .ΝΕΤ framework 3.1 runtimes for your version of Windows (you can download this from [here](https://dotnet.microsoft.com/download)).

## Citations

If you use any of this repository contents, please cite the following work:

	@inproceedings{zwicklbauer2020video,
	title={A fast smart-cropping method and dataset for video retargeting},
	author={Apostolidis, Konstantinos and Mezaris, Vasileios},
	booktitle={Proceedings of the 3rd International Workshop on AI for Smart TV Content Production, Access and Delivery},
	pages={17--24},
	year={2021}
	}

The original videos are taken from the [DHF1k](https://github.com/wenguanwang/DHF1K) dataset, therefore if you use these videos along with our annotations, please also cite the following work:

	@article{wang2019revisiting,
	title={Revisiting video saliency prediction in the deep learning era},
	author={Wang, Wenguan and Shen, Jianbing and Xie, Jianwen and Cheng, Ming-Ming and Ling, Haibin and Borji, Ali},
	journal={IEEE transactions on pattern analysis and machine intelligence},
	volume={43},
	number={1},
	pages={220--237},
	year={2019},
	publisher={IEEE}
	}

