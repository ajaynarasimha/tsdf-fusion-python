# Volumetric TSDF Fusion of RGB-D Images in Python

<img src="images/fusion-movie.gif" height=250px align="right"/>

This is a lightweight python script that fuses multiple registered color and depth images into a projective truncated signed distance function (TSDF) volume, which can then be used to create high quality 3D surface meshes and point clouds. Tested on Ubuntu 16.04.

An older CUDA/C++ version can be found [here](https://github.com/andyzeng/tsdf-fusion).

## Requirements

* Python 2.7+ with [NumPy](http://www.numpy.org/), [PyCUDA](https://developer.nvidia.com/pycuda), [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html), [Scikit-image](https://scikit-image.org/) and [Numba](https://numba.pydata.org/). These can be quickly installed/updated by running the following:
  ```shell
  pip install --user numpy opencv-python scikit-image numba
  ```
* [Optional] GPU acceleration requires an NVIDA GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyCUDA](https://developer.nvidia.com/pycuda):
  ```shell
  pip install --user pycuda
  ```
* [Optional] [OPENCL](https://www.khronos.org/opencl) acceleration and [PyOPENCL](https://documen.tician.de/pyopencl/):
  ```shell
  pip install --user pyopencl
  ```
* Installation using Anaconda:
  * OPENCL
    ```shell
    conda env create -f environment_opencl.yml
    ```
  * CUDA
    ```shell
    conda env create -f environment_cuda.yml
    ```
    
## Demo

This demo fuses 1000 RGB-D images from the 7-scenes dataset into a 405 x 264 x 289 projective TSDF voxel volume with 2cm resolution at about 30 FPS in GPU mode (0.4 FPS in CPU mode), and outputs a 3D mesh `mesh.ply` which can be visualized with a 3D viewer like [Meshlab](http://www.meshlab.net/).

**Note**: color images are saved as 24-bit PNG RGB, depth images are saved as 16-bit PNG in millimeters.

```shell
python demo.py
```

## Seen In
 * [Semantic Scene Completion from a Single Depth Image (CVPR 2017)](http://sscnet.cs.princeton.edu/)
 * [Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images (CVPR 2016)](http://dss.cs.princeton.edu/)

## References
 * [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions (CVPR 2017)](http://3dmatch.cs.princeton.edu/)
 * [A Volumetric Method for Building Complex Models from Range Images (SIGGRAPH 1996)](https://graphics.stanford.edu/papers/volrange/volrange.pdf)
 * [KinectFusion: Real-Time Dense Surface Mapping and Tracking (ISMAR 2011)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf)
 * [Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images (CVPR 2013)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RelocForests.pdf)
