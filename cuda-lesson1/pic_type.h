// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************
#include <opencv2/opencv.hpp>

#pragma once

// Structure definition for exchanging data between Host and Device
struct CudaPic{
    uint3 m_size;
    void * data;

    CudaPic(cv::Mat * src){
        m_size.x = src->cols;
        m_size.y = src->rows;
        data = src->data;
    }

    template < typename T>
    __device__ T  &at(int y, int x){
        T * return_data = (T *) data;
        return return_data[y * m_size.x + x];
    } 

};
