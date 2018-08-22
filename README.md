# Pretrained models for Moments in Time Dataset

We release the pre-trained models trained on [Moments in Time](http://moments.csail.mit.edu/).


### Models

* RGB model in PyTorch (ResNet50 pretrained on ImageNet). Run the following [script](test_model.py) to download and run the test sample. The model is tested sucessfully in PyTorch0.3 + python36. 
```
    python test_model.py
```
To test the model on your own video, supply the path of an mp4 file to this [script](test_video.py) like so:
```
    python test_video.py --video_file path_to_video/
```
In this case,

```
python test_video.py --video_file parent_dir/

```

