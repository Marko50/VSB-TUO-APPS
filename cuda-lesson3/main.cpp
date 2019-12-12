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
#include <sys/time.h>

#include "uni_mem_allocator.h"
#include "pic_type.h"
#include "animation.h"


// Function prototype from .cu file
void cu_create_chessboard( CudaPic t_pic, int t_square_size );
void cu_create_alphaimg( CudaPic t_pic, uchar3 t_color );
void cu_insertimage( CudaPic t_big_pic, CudaPic t_small_pic, int2 t_position );
void cu_bilin_scale(CudaPic ori, CudaPic dest, int t_square_size);
void cu_rotate(CudaPic ori, CudaPic dest, float2 angle_props, int t_square_size );

void resize_action( cv::Mat& t_orig){
    cv::Mat l_resize_down( t_orig.rows/4, t_orig.cols/4, CV_8UC4 );
    CudaPic cuda_resize_down(&l_resize_down);
    CudaPic original(&t_orig);
    cu_bilin_scale( original, cuda_resize_down,  32);
    cv::imshow( "Resize down", l_resize_down );
}

int main( int t_numarg, char **t_arg )
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	// some argument?
	if ( t_numarg > 2 ){
		// Load image
        cv::Mat snowflake = cv::imread( t_arg[ 2 ], CV_LOAD_IMAGE_UNCHANGED );
        
        if ( !snowflake.data )
			printf( "Unable to read file '%s'\n", t_arg[ 2 ] );

        else if (snowflake.channels() != 4 )
            printf( "Snowflake does not contain alpha channel!\n" );

		else{
            cv::Mat snowflake_animation = cv::imread(t_arg[1], cv::IMREAD_COLOR);
            if ( !snowflake_animation.data ){
                printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
                return -1;
            }

            cv::Mat snowflake_resized( snowflake.rows/4, snowflake.cols/4, CV_8UC4 );
            CudaPic cuda_snowflake_resized(&snowflake_resized);
            CudaPic cuda_snowflake(&snowflake);
            cu_bilin_scale( cuda_snowflake, cuda_snowflake_resized,  32);

            
            //( 777, 888, CV_8UC3 );

            Animation l_animation;
            CudaPic cuda_snow_animation(&snowflake_animation);

            l_animation.start( cuda_snow_animation, cuda_snowflake_resized );
            
            int posx = snowflake_animation.cols/2;
            int posy = 0;
            int angle = 0;
            float velocity = 200.0; 
            float angular_velocity = 500.0; 

            timeval l_start_time, l_cur_time, l_old_time, l_delta_time;
            gettimeofday( &l_old_time, NULL );
            l_start_time = l_old_time;

            CudaPic cuda_snowflake_rotated;
            cv::Mat snowflake_rotated;
            
            while (1){
                
                cv::waitKey( 1 );
                gettimeofday( &l_cur_time, NULL );
            	timersub( &l_cur_time, &l_old_time, &l_delta_time );
            	if ( l_delta_time.tv_usec < 1000 ) continue; // too short time
            	l_old_time = l_cur_time;
                float delta_secs = l_delta_time.tv_usec / 1E6;
                posy +=  (int) velocity*delta_secs;
                angle += (int) angular_velocity*delta_secs;
                if (posy + snowflake_resized.rows > snowflake_animation.rows){
                    posy = snowflake_animation.rows;
                    break;
                }
                 //{ sinf(angle*3.14/180), cosf(angle*3.14/180)}
                int new_cols = snowflake_resized.cols;
                int new_rows = snowflake_resized.rows;
                if (angle < 90)
                {
                    int new_cols = (snowflake_resized.cols*cosf(angle*3.14/180)) + (snowflake_resized.rows*sinf(angle*3.14/180));
                    int new_rows = (snowflake_resized.cols*sinf(angle*3.14/180)) + (snowflake_resized.rows*cosf(angle*3.14/180));
                }
                else if(angle == 90){
                    new_cols = snowflake_resized.rows;
                    new_rows = snowflake_resized.cols;
                }
                else{
                    int num_times = angle/90;
                    int aux_angle = aux_angle - 90*num_times;
                    int new_cols = (snowflake_resized.rows*cosf(aux_angle*3.14/180)) + (snowflake_resized.cols*sinf(aux_angle*3.14/180));
                    int new_rows = (snowflake_resized.rows*sinf(aux_angle*3.14/180)) + (snowflake_resized.cols*cosf(aux_angle*3.14/180));
                }
                snowflake_rotated = cv::Mat( new_rows, new_cols, CV_8UC4 );
                cuda_snowflake_rotated = CudaPic(&snowflake_rotated);
                cuda_snowflake = CudaPic(&snowflake_resized);
                cu_rotate( cuda_snowflake, cuda_snowflake_rotated, { sinf(angle*3.14/180), cosf(angle*3.14/180)}, 1);
                l_animation.m_cuda_ins_pic = cuda_snowflake_rotated;
                l_animation.next(cuda_snow_animation, {posx,posy});
                cv::imshow( "Animation", snowflake_animation );
            }
            printf("End of animation...\r\n");
            cv::waitKey( 0 );
		}
	}
    else{
        printf( "Usage: ./main background background snowflake \n");
    }
	
}