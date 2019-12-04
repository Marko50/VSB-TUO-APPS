// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
//#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "pic_type.h"

// Function prototype from .cu file
void cu_create_chessboard( CudaPic t_pic, int t_square_size );
void cu_create_alphaimg( CudaPic t_pic, uchar3 t_color );
void cu_insertimage( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position );
void cu_bilin_scale(CudaPic ori, CudaPic dest, int t_square_size);
void cu_rotate(CudaPic ori, CudaPic dest, float2 angle_props, int t_square_size );

void resize_action( cv::Mat& t_orig){
    cv::Mat l_resize_up( t_orig.rows * 2, t_orig.cols / 2 * 3, CV_8UC3 );
    cv::Mat l_resize_down( t_orig.rows  / 2, t_orig.cols / 2, CV_8UC3 );
    CudaPic cuda_resize_up(&l_resize_up);
    CudaPic cuda_resize_down(&l_resize_down);
    CudaPic original(&t_orig);
    cu_bilin_scale( original, cuda_resize_up,  32);
    cu_bilin_scale( original, cuda_resize_down,  32);
    cv::imshow( "Resize up", l_resize_up );
    cv::imshow( "Resize down", l_resize_down );
}

void rotate_action(cv::Mat& t_orig, float t_alpha){
    int l_diagonal = sqrtf( t_orig.cols * t_orig.cols + t_orig.rows * t_orig.rows );
    cv::Mat l_rotate( l_diagonal, l_diagonal, CV_8UC3 );
    float t_sin = sinf( t_alpha );
    float t_cos = cosf( t_alpha );
    CudaPic cuda_rotate(&l_rotate);
    CudaPic original(&t_orig);
    cu_rotate(original,cuda_rotate, {t_sin, t_cos}, 1);
    cv::imshow( "Rotated", l_rotate );
}

int main( int t_numarg, char **t_arg )
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	// some argument?
	if ( t_numarg > 1 ){
		// Load image
		cv::Mat l_cv_img = cv::imread( t_arg[ 1 ], CV_LOAD_IMAGE_UNCHANGED );

		if ( !l_cv_img.data )
			printf( "Unable to read file '%s'\n", t_arg[ 1 ] );

		else{
            cv::imshow( "Original", l_cv_img );
			resize_action(l_cv_img);
            rotate_action(l_cv_img, 3.14/4);
            cv::waitKey( 0 );
		}
	}
    else{
        printf( "Usage: ./main file_name \n");
    }
	
}

