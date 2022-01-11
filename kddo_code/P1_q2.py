"""
Created on Tue Oct  5 23:04:43 2021
Author: Khoa Dang Do
Instructor: Dr. Tianfu Wu
ECE 558-01 (Fall 2021)
Project 1: Part 2 - 2D FFT & iFFT
**License** © 2021 - 2021 Khoa Do. All rights reserved.
"""










""" import necessary packages """
import numpy as np
import cv2
import matplotlib.pyplot as mpl









""" 2-D FFT function """
def DFT2(f):

    f_2D = np.zeros(f.shape, dtype=complex)                     # create image same size as the original image, filled with 0's

    """ 2-D FFT algorithm """
    for i in range(f.shape[0]):
        f_2D[i, :] = np.fft.fft(f[i, :])                        # do 1-D FFT on rows of original image
    for i in range(f.shape[1]):                                 # do 1-D FFT on columns of img agter being 1-D FFT-ed 
        f_2D[:, i] = np.fft.fft(f_2D[:, i])

    return f_2D                                                 # return transformed image


""" 2-D iFFT function """
def IDFT2(F):
   
    F_I2D = np.zeros(F.shape, dtype=complex)                    # create image same size as the transformed image, filled with 0's

    for i in range(F.shape[0]):                                 # do 1-D iFFT on rows of transformed image
        F_I2D[i, :] = np.fft.ifft(F[i, :])  
    for i in range(F.shape[1]):                                 # do 1-D FFT on columns of img agter being 1-D iFFT-ed
        F_I2D[:, i] = np.fft.ifft(F_I2D[:, i])

    return F_I2D                                                # return transformed image










def main_function():
    
    
    prompt1 = input("Choose 'lena.png' or 'wolves.png': ")
    while (prompt1 != "lena.png" and prompt1 != "wolves.png"):
        prompt1 = input("Please type 'lena.png' or 'wolves.png': ")
        
        
    f = cv2.imread(prompt1,0)/255                               # read img as grayscale then scale to [0,1] by / 255
    
    
    """ Part 2a """
    F = DFT2(f)                                                 # call DFT2 to do 2-D FFT
    
    
    """ visualizing transformed image's spectrum and phase angle """
    s = np.log(1 + np.absolute(F))                              
    s_shift = np.log(1+np.abs(np.fft.fftshift(F)))              # centering s
    phase_angle = np.angle(F)
    
    
    """ show figures """
    mpl.figure(1)
    mpl.imshow(s)
    mpl.figure(2)
    mpl.imshow(s_shift)
    mpl.figure(3)
    mpl.imshow(phase_angle)
    
    
    """ Part 2b """
    g = IDFT2(F)                                                # call IDFT2 to do 2-D iFFT
    
    d = np.abs(f - np.abs(g))                                   # check if original img f - magitude(iFFT) = 0 = black image
                                                                #   take abs of f - magitude(iFFT) to avoid negative values
    
    cv2.imwrite('Part_2b_img-iFFT.png', d)                # write image back as Part_2a_img-iFFT.png
    









if __name__ == "__main__":
    main_function()