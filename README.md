
# RetargetVid & SmartVidCrop

## Introduction
This repository contains **RetargetVid** - a video retargeting dataset, and **SmartVidCrop** - a fast cropping method for video retargeting.

We observed that each literature work about video retargeting uses an arbitrary selection of videos to test its results, and most of the time these videos are not provided, while the evaluation procedure relies on visual inspection of selected frames. Motivated by this we construct and release RetargetVid, a  publicly available benchmark dataset for video cropping, annotated by 6 human subjects.

We also release a new, rather simple, yet fast and well-performing, video cropping method, which selects the main focus out of the multiple possible salient regions of the video by introducing a new filtering-through-clustering processing step.

## RetargetVid Dataset
We selected a subset of 200 videos from the publicly available videos of the [DHF1k](https://github.com/wenguanwang/DHF1K) dataset, specifically, the first 100 videos of the training set (videos with indices 001 to 100) and the 100 videos of the validation set (videos with indices 601 to 700). All videos are in 16:9 aspect ratio and most of them consist of a single shot.

We invited 6 human subjects and asked them to select the region of each frame that would be ideal to be included in a cropped version of the video. Specifically, we assigned them the task of generating two cropped versions for each video, one with target aspect ratio of 1:3 and another one with target aspect ratio of 3:1. We selected these extreme target aspect ratios (despite not being used in real-life applications) in order to identify human preferences under very demanding circumstances. Moreover, less extreme target aspect ratios can still be evaluated by assessing to what extent an e.g. 9:16 crop window includes the 1:3 manually specified window.

Our crop window annotations for each video are in the form of text files, where the *i*-th line contains the top-left coordinates of the crop window for the *i*-th frame. The coordinates are zero-based, so the top-left pixel of a frame is the (0,0). There are 2400 annotation text files in total (200 videos * 2 target aspect ratios * 6 annotator subjects). The annotation text files are named *$video_id$-$target_aspect ratio$.txt*, where *$video_id$* is the original video filename and $target_aspect ratio$ is the target aspect ratio (i.e. "1-3" or "3-1"). The annotation text files can be found in the *annotations* directory of this repository, where a separate zip file is provided for each annotator. To download the actual DHF1k videos, follow the download links in the original author's GitHub repository [here](https://github.com/wenguanwang/DHF1K)


## SmartVidCrop Method
We argue that cropping methods are more suitable for video aspect ratio transformation when the minimization of semantic distortions is a prerequisite.  Therefore, we present a new, rather simple, yet fast and well-performing, video cropping method, which selects the main focus out of the multiple possible salient regions of the video by introducing a new filtering-through-clustering processing step. For our method, we utilize visual saliency to find the image regions of attention, and we employ a filtering-through-clustering technique to select the main region of focus. For more details, see the first citation in Citations section.

Our method is implemented in Python 3 and the source code is available in the *SmartVidCrop* directory of this repository.


## Evaluation
To evaluate the results of your method with respect to the ground truth annotations of the **RetargetVid** dataset, first download this repository. Then, create a new sub-directory in the *results* directory, in which a text file for each of the 200 videos of the dataset must be created. The files must follow the naming convension *$video_id$-$target_aspect ratio$.txt*, where *$video_id$* is the original video filename and $target_aspect ratio$ is the target aspect ratio (i.e. "1-3" or "3-1"). Each line of this text file must have the crop window (top, left, bottom, right) coordinates. Finally, run the *retargetvid_eval.py* python script. The evaluation results for every sub-directory in the *results* directory will be displayed, warning you if there were any errors in the process or any incomplete annotations were found.

The evaluation results are calculated as the mean similarity of all crop windows between the results contained in a sub-directory of the *results* directory and the RetargetVid dataset's ground-truth annotations. The similarity is calculated in terms of the Intersection over Union (IoU) scores.

In the *results* directory we include two sub-directories:
* *smartvidcrop* with the results of our method, and
* *autoflip* with the results of Google's [AutoFlip](https://google.github.io/mediapipe/solutions/autoflip) method,
for you to quickly replicate the results of our paper (see the first citation in Citations section).
 
The software was implemented in Python 3 and the source code is included in the *retargetvid_eval.py* file of this repository.




## Annotator Software
To assist the annotators in their task we implemented a graphical user interface tool which facilitates the navigation throughout the video, allows the user to set a crop window for each frame through simple drag-and-drop mouse operations, and overlays the crop window on the video frames to allow for the quick inspection of the user's decisions.

The software was implemented in C# and the source code is available in the *annotator_software* directory of this repository. You will need Visual Studio 2019 to compile the source code.



## Prerequisities
To run our SmartVidCrop method you will need Python 3. You must also have the following packages which can be simply installed via pip (in parenthesis you can find the recommended version to install):

* TensorFlow (tensorflow-gpu==2.0)
* PyTorch (torch==1.7.1+cu110)
* hdbscan (hdbscan==0.8.26)
* ffmpeg (ffmpeg==1.4)
* OpenCV (opencv-python==4.2.0)
* SciPy (scipy==1.5.1)
* ImUtils (imutils==0.5.4)
* Scikit-learn (scikit-learn-0.24.1)

SmartVidCrop also uses the following libraries
* [Unisal](https://github.com/rdroste/unisal)
* [TransNetV2](https://github.com/soCzech/TransNetV2)

but these packages are already included in the 3rd_party_libs directory.


To run the evaluator software you will just need Python 3.

To run the annotator software you will need .ΝΕΤ framework 3.1 runtimes for your version of Windows (you can download this from [here](https://dotnet.microsoft.com/download)).

## Citations

If you use any of this repository contents, please cite the following work:
```
@inproceedings{kapost2021afast,
title={A fast smart-cropping method and dataset for video retargeting},
author={Apostolidis, Konstantinos and Mezaris, Vasileios},
booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
year={2021},
organization={IEEE}
}
```

The original videos are taken from the [DHF1k](https://github.com/wenguanwang/DHF1K) dataset, which was introduced in the following work:

```
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
```


