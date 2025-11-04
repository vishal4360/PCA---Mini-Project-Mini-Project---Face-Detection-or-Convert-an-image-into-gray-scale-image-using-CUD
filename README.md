[[# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming
# CUDA Grayscale Conversion

## Aim:
The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:
1. Load the input image using the `stb_image` library.
2. Allocate memory on the GPU for the input and output image buffers.
3. Copy the input image data from the CPU to the GPU.
4. Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.
5. Launch the CUDA kernel with appropriate grid and block dimensions.
6. Copy the resulting grayscale image data from the GPU back to the CPU.
7. Save the grayscale image using the `stb_image_write` library.
8. Clean up allocated memory.
   
## Program:
```python
!apt-get update
!apt-get install -y build-essential cmake pkg-config libopencv-dev

%%writefile grayscale.cu
#define CHANNELS 3
#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

__global__
void colorConvertToGrey(unsigned char *rgb, unsigned char *grey, int rows, int cols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows)
    {
        int grey_offset = row * cols + col;
        int rgb_offset = grey_offset * CHANNELS;

        unsigned char r = rgb[rgb_offset + 0];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char b = rgb[rgb_offset + 2];

        grey[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}

size_t loadImageFile(unsigned char **h_rgb_image, const std::string &input_file, int *rows, int *cols)
{
    cv::Mat img_data = cv::imread(input_file.c_str(), cv::IMREAD_COLOR);
    if (img_data.empty())
    {
        std::cerr << "Unable to load image file: " << input_file << std::endl;
        exit(1);
    }

    *rows = img_data.rows;
    *cols = img_data.cols;

    size_t num_pixels = (*rows) * (*cols);
    *h_rgb_image = (unsigned char *)malloc(num_pixels * CHANNELS);
    memcpy(*h_rgb_image, img_data.data, num_pixels * CHANNELS);

    return num_pixels;
}

void outputImage(const std::string &output_file, unsigned char *grey_image, int rows, int cols)
{
    cv::Mat greyData(rows, cols, CV_8UC1, grey_image);
    cv::imwrite(output_file.c_str(), greyData);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: <executable> input_file output_file" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    unsigned char *h_rgb_image, *h_grey_image;
    unsigned char *d_rgb_image, *d_grey_image;
    int rows, cols;

    const size_t total_pixels = loadImageFile(&h_rgb_image, input_file, &rows, &cols);
    h_grey_image = (unsigned char *)malloc(sizeof(unsigned char) * total_pixels);

    cudaMalloc(&d_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS);
    cudaMalloc(&d_grey_image, sizeof(unsigned char) * total_pixels);
    cudaMemcpy(d_rgb_image, h_rgb_image, sizeof(unsigned char) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    colorConvertToGrey<<<grid, block>>>(d_rgb_image, d_grey_image, rows, cols);

    cudaMemcpy(h_grey_image, d_grey_image, sizeof(unsigned char) * total_pixels, cudaMemcpyDeviceToHost);
    outputImage(output_file, h_grey_image, rows, cols);

    cudaFree(d_rgb_image);
    cudaFree(d_grey_image);
    free(h_rgb_image);
    free(h_grey_image);

    return 0;
}

!nvcc grayscale.cu -o grayscale `pkg-config --cflags --libs opencv4`

from google.colab import files
uploaded = files.upload()
input_file = list(uploaded.keys())[0]
!./grayscale {input_file} output.jpg

import cv2
import numpy as np
import matplotlib.pyplot as plt

original = cv2.imread("/content/1122730.jpg")
gray = cv2.imread("/content/gray_output.jpg", cv2.IMREAD_GRAYSCALE)

if original is None or gray is None:
    raise FileNotFoundError("Image not found! Check the filename or path.")

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
gray = np.array(gray, dtype=np.uint8)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Colour Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.tight_layout()
plt.show()
```
## Output:
![WhatsApp Image 2025-11-04 at 13 47 39_4f0ab9cc](https://github.com/user-attachments/assets/ff47ed96-c736-4480-9a52-98c1d3ed2fbc)


## Result:
The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
](https://github.com/vishal4360/PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24/blob/main/README.md)](https://github.com/vishal4360/PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24/blob/main/README.md)
