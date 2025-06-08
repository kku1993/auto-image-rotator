# Automatic, Lossless JPEG Rotator

## Change note

- This fork implements lossless JPEG transformation and removes png support.
- Changes the command line signature
- Removed docker setup

## Overview
This app uses the OpenCV and Dlib computer vision libraries to auto rotate images based on detected human faces.

*This is useful for auto rotating images in bulk that do not contain EXIF orientation meta data (e.g., scanned photos).*

Currently, this is only effective for images that contain one or more face. In the future, [advanced CNN techniques](https://d4nst.github.io/2017/01/12/image-orientation/) could be implemented to auto correct the rotation for any photo.

![title](images/example-before-after.jpg)

## Setup

#### Python venv

```
apt-get update
apt-get install cmake libjpeg8-dev libturbojpeg-dev
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Usage
After the one-time setup, rotating a directory of images is as simple as running this command:

```
python3 rotate.py {IMAGE_PATH}
```

By default, rotated images are saved as new files with a `*-rotated` filename pattern in your `IMAGES_PATH` directory. If you're comfortable overwriting your original files with rotated versions you may use the `--overwrite=true` param.
