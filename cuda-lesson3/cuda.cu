// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage wit unified memory.
// Image transformation from RGB to BW schema. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "pic_type.h"
#include "animation.h"


__global__ void kernel_rotate(CudaPic ori, CudaPic dest, float2 angle_props){
	// X,Y coordinates and check image dimensions
	int l_rotated_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_rotated_x = blockDim.x * blockIdx.x + threadIdx.y;
	
	if ( l_rotated_y >= dest.m_size.y ) return;
	if ( l_rotated_x >= dest.m_size.x ) return;
	
	float t_sin = angle_props.x;
	float t_cos = angle_props.y;

	int l_crotate_x = l_rotated_x - dest.m_size.x / 2;
	int l_crotate_y = l_rotated_y - dest.m_size.y / 2;

	// position in orig image
	float l_corig_x = t_cos * l_crotate_x - t_sin * l_crotate_y;
	float l_corig_y = t_sin * l_crotate_x + t_cos * l_crotate_y;
	// recalculation from centerpoint coordinates to image coordinates
	int l_orig_x = l_corig_x + ori.m_size.x / 2;
	int l_orig_y = l_corig_y + ori.m_size.y / 2;
	// out of orig image?
	if ( l_orig_y >= ori.m_size.y || l_orig_y < 0) return;
	if ( l_orig_x >= ori.m_size.x || l_orig_x < 0 ) return;

	dest.at<uchar4>( l_rotated_y, l_rotated_x ) = ori.at<uchar4>( l_orig_y, l_orig_x );
}

void cu_rotate(CudaPic ori, CudaPic dest, float2 angle_props, int t_square_size ){
	cudaError_t l_cerr;
	
	// Grid creation, size of grid must be equal or greater than images
	dim3 l_blocks( ( dest.m_size.x + t_square_size - 1 ) / t_square_size,
			       ( dest.m_size.y + t_square_size - 1 ) / t_square_size );
	dim3 l_threads( t_square_size, t_square_size );
	kernel_rotate<<< l_blocks, l_threads >>>( ori, dest, angle_props );
	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

__global__ void kernel_bilin_scale(CudaPic ori, CudaPic dest){
	// X,Y coordinates and check image dimensions
	int l_resize_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_resize_x = blockDim.x * blockIdx.x + threadIdx.x;
	
	if ( l_resize_y >= dest.m_size.y ) return;
	if ( l_resize_x >= dest.m_size.x ) return;

	float l_scale_x = ori.m_size.x - 1;
	float l_scale_y = ori.m_size.y - 1;
	
	l_scale_x /= dest.m_size.x;
	l_scale_y /= dest.m_size.y;

	// new real position
	float l_orig_x = l_resize_x * l_scale_x;
	float l_orig_y = l_resize_y * l_scale_y;

	if ( l_orig_y >= ori.m_size.y || l_orig_y < 0) return;
	if ( l_orig_x >= ori.m_size.x || l_orig_x < 0 ) return;

	// diff x and y
	float l_diff_x = l_orig_x - ( int ) l_orig_x;
	float l_diff_y = l_orig_y - ( int ) l_orig_y;
	
	// points
	uchar4 bgr00 = ori.at<uchar4>( ( int ) l_orig_y, ( int ) l_orig_x );
	uchar4 bgr01 = ori.at<uchar4>( ( int ) l_orig_y, 1 + ( int ) l_orig_x );
	uchar4 bgr10 = ori.at<uchar4>( 1 + ( int ) l_orig_y, ( int ) l_orig_x );
	uchar4 bgr11 = ori.at<uchar4>( 1 + ( int ) l_orig_y, 1 + ( int ) l_orig_x );
	
	uchar4 bgr;
	bgr.x = bgr00.x * ( 1 - l_diff_y ) * ( 1 - l_diff_x ) + bgr01.x * ( 1 - l_diff_y ) * ( l_diff_x ) + bgr10.x * ( l_diff_y ) * ( 1 - l_diff_x ) + bgr11.x * ( l_diff_y ) * ( l_diff_x );
	bgr.y = bgr00.y * ( 1 - l_diff_y ) * ( 1 - l_diff_x ) + bgr01.y * ( 1 - l_diff_y ) * ( l_diff_x ) + bgr10.y * ( l_diff_y ) * ( 1 - l_diff_x ) + bgr11.y * ( l_diff_y ) * ( l_diff_x );
	bgr.z = bgr00.z * ( 1 - l_diff_y ) * ( 1 - l_diff_x ) + bgr01.z * ( 1 - l_diff_y ) * ( l_diff_x ) + bgr10.z * ( l_diff_y ) * ( 1 - l_diff_x ) + bgr11.z * ( l_diff_y ) * ( l_diff_x );
	bgr.w = bgr00.w * ( 1 - l_diff_y ) * ( 1 - l_diff_x ) + bgr01.w * ( 1 - l_diff_y ) * ( l_diff_x ) + bgr10.w * ( l_diff_y ) * ( 1 - l_diff_x ) + bgr11.w * ( l_diff_y ) * ( l_diff_x );
	
	dest.at<uchar4>(l_resize_y ,  l_resize_x) = bgr;
}

void cu_bilin_scale( CudaPic ori, CudaPic dest,  int t_square_size )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	dim3 l_blocks( ( dest.m_size.x + t_square_size - 1 ) / t_square_size,
			       ( dest.m_size.y + t_square_size - 1 ) / t_square_size );
	dim3 l_threads( t_square_size, t_square_size );
	kernel_bilin_scale<<< l_blocks, l_threads >>>( ori, dest );
	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

// Demo kernel to create chess board
__global__ void kernel_chessboard( CudaPic t_color_pic )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;

	unsigned char b_or_w = 255 * ( ( blockIdx.x + blockIdx.y ) & 1 );

	// Store point into image
	t_color_pic.at<uchar3>(l_y, l_x) =  { b_or_w, b_or_w, b_or_w };
}

void cu_create_chessboard( CudaPic t_color_pic, int t_square_size )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	dim3 l_blocks( ( t_color_pic.m_size.x + t_square_size - 1 ) / t_square_size,
			       ( t_color_pic.m_size.y + t_square_size - 1 ) / t_square_size );
	dim3 l_threads( t_square_size, t_square_size );
	kernel_chessboard<<< l_blocks, l_threads >>>( t_color_pic );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------------------------

//Demo kernel to create picture with alpha channel gradient
__global__ void kernel_alphaimg( CudaPic t_color_pic, uchar3 t_color )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;

	int l_diagonal = sqrtf( t_color_pic.m_size.x * t_color_pic.m_size.x + t_color_pic.m_size.y * t_color_pic.m_size.y );
	int l_dx = l_x - t_color_pic.m_size.x / 2;
	int l_dy = l_y - t_color_pic.m_size.y / 2;
	int l_dxy = sqrtf( l_dx * l_dx + l_dy * l_dy ) - l_diagonal / 2;

	// Store point into image
	t_color_pic.at<uchar4>( l_y ,l_x ) =
		{ t_color.x, t_color.y, t_color.z, ( unsigned char ) ( 255 - 255 * l_dxy / ( l_diagonal / 2 ) ) };
}

void cu_create_alphaimg( CudaPic t_color_pic, uchar3 t_color )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_color_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_color_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_alphaimg<<< l_blocks, l_threads >>>( t_color_pic, t_color );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------------------------

// Demo kernel to create picture with alpha channel gradient
__global__ void kernel_insertimage( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_small_pic.m_size.y ) return;
	if ( l_x >= t_small_pic.m_size.x ) return;
	int l_by = l_y + t_position.y;
	int l_bx = l_x + t_position.x;
	if ( l_by >= t_big_pic.m_size.y || l_by < 0 ) return;
	if ( l_bx >= t_big_pic.m_size.x || l_bx < 0 ) return;

	// Get point from small image
	uchar4 l_fg_bgra = t_small_pic.at<uchar4>( l_y , l_x );
	uchar3 l_bg_bgr = t_big_pic.at<uchar3>( l_by , l_bx );
	uchar3 l_bgr = { 0, 0, 0};

	// compose point from small and big image according alpha channel
	l_bgr.x = l_fg_bgra.x * l_fg_bgra.w / 255 + l_bg_bgr.x * ( 255 - l_fg_bgra.w ) / 255;
	l_bgr.y = l_fg_bgra.y * l_fg_bgra.w / 255 + l_bg_bgr.y * ( 255 - l_fg_bgra.w ) / 255;
	l_bgr.z = l_fg_bgra.z * l_fg_bgra.w / 255 + l_bg_bgr.z * ( 255 - l_fg_bgra.w ) / 255;

	// Store point into image
	t_big_pic.at<uchar3>( l_by ,l_bx ) = l_bgr;
}

void cu_insertimage( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 32;
	dim3 l_blocks( ( t_small_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( t_small_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insertimage<<< l_blocks, l_threads >>>( t_big_pic, t_small_pic, t_position );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}

__global__ void kernel_create_image( CudaPic t_color_pic, CudaPic pic )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;

	// Store point into image
	t_color_pic.at<uchar3>( l_y , l_x ) = pic.at<uchar3>(l_y, l_x);
}


void Animation::start( CudaPic t_bg_pic, CudaPic t_ins_pic )
{
	if ( m_initialized ) return;
	cudaError_t l_cerr;

	m_cuda_bg_pic = t_bg_pic;
	m_cuda_res_pic = t_bg_pic;
	m_cuda_ins_pic = t_ins_pic;

	// Memory allocation in GPU device
	// Memory for background
	l_cerr = cudaMalloc( &m_cuda_bg_pic.data, m_cuda_bg_pic.m_size.x * m_cuda_bg_pic.m_size.y * sizeof( uchar3 ) );
	if ( l_cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Creation of background gradient
	int l_block_size = 32;
	dim3 l_blocks( ( m_cuda_bg_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( m_cuda_bg_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_create_image<<< l_blocks, l_threads >>>( m_cuda_bg_pic, t_bg_pic );

	m_initialized = 1;
}

void Animation::next( CudaPic t_res_pic, int2 t_position )
{
	if ( !m_initialized ) return;

	cudaError_t cerr;

	// Copy data internally GPU from background into result
	cerr = cudaMemcpy( m_cuda_res_pic.data, m_cuda_bg_pic.data, m_cuda_bg_pic.m_size.x * m_cuda_bg_pic.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// insert picture
	int l_block_size = 1;
	dim3 l_blocks( ( m_cuda_ins_pic.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( m_cuda_ins_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );

	kernel_insertimage<<< l_blocks, l_threads >>>( m_cuda_res_pic, m_cuda_ins_pic, t_position );

	// Copy data to GPU device
	cerr = cudaMemcpy( t_res_pic.data, m_cuda_res_pic.data, m_cuda_res_pic.m_size.x * m_cuda_res_pic.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

}

void Animation::stop()
{
	if ( !m_initialized ) return;

	cudaFree( m_cuda_bg_pic.data );
	cudaFree( m_cuda_res_pic.data );
	cudaFree( m_cuda_ins_pic.data );

	m_initialized = 0;
}