# 2D-Convolution-FFT-iFFT

## Type: Academic Individual Project

## Project Description
NCSU ECE 558 (Digital Imaging Systems) Project 1
  - Implement 2D convolution, discrete fast Fourier transform, and inverse discrete fast Fourier transform functions from scratch in Python

## Dependencies
  - numpy
  - cv2
  - Spyder IDE (used), Linux, or Anaconda on Windows 10 Education
  
## About the Repo.
  - kddo_code contains **P1_q1.py** and **P2_q2.py** that are carefully commented 
    - **P1_q1.py** implements the 2-D convolution function.  It convolves an input image of choice from the user (grayscale or RGB) with a filter of choice (box filter, Sobel Mx, etc). Padding function algorithm is implemted without the use of Python built-in function that directly performs zero/wrap-around/copy-edge/reflect-across-edge padding
    - **P2_q2.py** implements the 2-D FFT and iFFT with the use of buit-in 1-D NumPy FFT/iFFT functions
  - kddo_images contains a folder of original images and a folder of result images
  - Final report details the implementations of the functions in this project along with the results + analysis

## Author
Khoa Do

## Reference
N/A

## Additional Notes
N/A
