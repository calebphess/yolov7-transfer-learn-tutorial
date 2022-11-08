# YOLOv7 Transfer Learning Tutorial
**A tutorial on how to transfer learn the YOLOv7 model on a new labeled dataset. It will also show you how to preprocess the data to meet the proper format to be trained.**

## Citations
- **[YOLOv7](https://github.com/WongKinYiu/yolov7) made all of this possible.**
- **Thank you [JJ Jordan](https://www.pexels.com/@see2believe/) for the royalty free test image! (data/test-image.jpg)**

## Prerequisites
- [Conda](https://www.anaconda.com/products/distribution) is installed on your machine
- If you are using GPU's to train (which is highly recommended) then ensure [cuda](https://developer.nvidia.com/cuda-toolkit) is configured properly

## Setup
- Create the conda environment
    - `conda create --name "yolov7-tutorial" python=3.6.9`
- Activate the conda envrinment
    - `conda activate yolov7-tutorial`
    - **NOTE:** you will need to do this every time you run a new terminal session
- Install the required pip packages
    - `pip install -r requirements.txt`
- Verify that YOLOv7 is working properly ()
    - `./scripts/test_yolo_image_detection.sh`
        - The response should look something like this `The image with the result is saved in: runs/detect/expX/test-image.jpg`
        - You can open that result and you should see a bounding box or two in the image showing detections
    - Optionally if you are on Mac or have a camera connected to your machine you can run the following to test video
        - `./scripts/test_yolo_video_detection.sh `
        - **NOTE:** you may have to accept the request to access the camera and run the command a second time
        - You can exit by hitting `ctrl+c` a couple times in the terminal window
        
## Troubleshooting
- "I'm getting some miscellaneous python error referring to a missing library"
    1. Ensure that you have activated your conda environment
        -   `conda activate yolov7-tutorial`
    2. Verify that pip and python are actually referencing the conda versions
        -   `which python` and `which pip` should both resolve to something like `PATH_TO_HOME/opt/anaconda3/envs/yolov7-tutorial/bin/python`
            - If this isn't the case you probably have something in `.bashrc` that is hard coding the path of these binaries
            - See if `which python3` or `which pip3` resolves to the anaconda path above, if so just replace commands above with those
    3. If it's a cuda/Nvidia error make sure nvidia drivers are up to date and cuda is installed
        - `nvidia-smi` should list your graphics card(s)
            - if not check [here](https://hasindu2008.github.io/f5c/docs/cuda-troubleshoot) to see if any of these suggestions fix your issue