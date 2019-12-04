// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 09/2019
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         OpenCV simulator of LCD
//
// **************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

void blur( cv::Mat &t_orig, cv::Mat &t_blur, float t_level )
{
    for ( int l_blur_x = 1; l_blur_x < t_blur.cols - 1; l_blur_x++ )
        for ( int l_blur_y = 1; l_blur_y < t_blur.rows - 1; l_blur_y++ )
        {
            // initialize sum
            cv::Vec3i l_bgr32 = { 0, 0, 0 };
            // loop for all neighbours
            for ( int nx = -1; nx <= 1; nx++ )
                for( int ny = -1; ny <= 1; ny++ )
                {
                    // pickup point from orig figure
                    cv::Vec3b l_bgr = t_orig.at<cv::Vec3b>( l_blur_y + ny, l_blur_x + nx );
                    // sum of r/g/b colors
                    for ( int b = 0; b < 3; b++ )
                    {
                        if ( !nx && !ny ) 
                            l_bgr32[ b ] += l_bgr[ b ];  
                        else 
                            l_bgr32[ b ] += l_bgr[ b ] * t_level;
                    }
                }
            // average
            l_bgr32 /= 1 + 8 * t_level;
            // put pixel into blur image
            t_blur.at<cv::Vec3b>( l_blur_y, l_blur_x ) = l_bgr32;
        }
}

void rotate_ok( cv::Mat &t_orig, cv::Mat &t_rotate, float t_alpha )
{
    float t_sin = sinf( t_alpha );
    float t_cos = cosf( t_alpha );

    for ( int l_rotate_x = 0; l_rotate_x < t_rotate.cols; l_rotate_x++ )
        for ( int l_rotate_y = 0; l_rotate_y < t_rotate.rows; l_rotate_y++ )
        {
            // recalculation from image coordinates to centerpoint coordinates
            int l_crotate_x = l_rotate_x - t_rotate.cols / 2;
            int l_crotate_y = l_rotate_y - t_rotate.rows / 2;

            // position in orig image
            float l_corig_x = t_cos * l_crotate_x - t_sin * l_crotate_y;
            float l_corig_y = t_sin * l_crotate_x + t_cos * l_crotate_y;
            // recalculation from centerpoint coordinates to image coordinates
            int l_orig_x = l_corig_x + t_orig.cols / 2;
            int l_orig_y = l_corig_y + t_orig.rows / 2;
            // out of orig image?
            if ( l_orig_x < 0 || l_orig_x >= t_orig.cols ) continue;
            if ( l_orig_y < 0 || l_orig_y >= t_orig.rows ) continue;

            t_rotate.at<cv::Vec3b>( l_rotate_y, l_rotate_x ) = t_orig.at<cv::Vec3b>( l_orig_y, l_orig_x );
        }
}
void rotate_bad( cv::Mat &t_orig, cv::Mat &t_rotate, float t_alpha )
{
    float t_sin = sinf( t_alpha );
    float t_cos = cosf( t_alpha );

    for ( int l_orig_x = 0; l_orig_x < t_orig.cols; l_orig_x++ )
        for ( int l_orig_y = 0; l_orig_y < t_orig.rows; l_orig_y++ )
        {
            // recalculation from image coordinates to centerpoint coordinates
            int l_corig_x = l_orig_x - t_orig.cols / 2;
            int l_corig_y = l_orig_y - t_orig.rows / 2;

            // position in orig image and
            // recalculation from centerpoint coordinates to image coordinates
            int l_rotate_x = t_cos * l_corig_x - t_sin * l_corig_y + t_rotate.cols / 2;
            int l_rotate_y = t_sin * l_corig_x + t_cos * l_corig_y + t_rotate.rows / 2;
            // out of rotated image?
            if ( l_rotate_x < 0 || l_rotate_x >= t_rotate.cols ) continue;
            if ( l_rotate_y < 0 || l_rotate_y >= t_rotate.rows ) continue;

            t_rotate.at<cv::Vec3b>( l_rotate_y, l_rotate_x ) = t_orig.at<cv::Vec3b>( l_orig_y, l_orig_x );
        }
}

void bilin_scale( cv::Mat &t_orig, cv::Mat &t_resize )
{
    float l_scale_x = t_orig.cols - 1;
    float l_scale_y = t_orig.rows - 1;
    l_scale_x /= t_resize.cols;
    l_scale_y /= t_resize.rows;

    for ( int l_resize_x = 0; l_resize_x < t_resize.cols; l_resize_x++ )
        for ( int l_resize_y = 0; l_resize_y < t_resize.rows; l_resize_y++ )
        {
            // new real position
            float l_orig_x = l_resize_x * l_scale_x;
            float l_orig_y = l_resize_y * l_scale_y;
            // diff x and y
            float l_diff_x = l_orig_x - ( int ) l_orig_x;
            float l_diff_y = l_orig_y - ( int ) l_orig_y;

            // points
            cv::Vec3b bgr00 = t_orig.at<cv::Vec3b>( ( int ) l_orig_y, ( int ) l_orig_x );
            cv::Vec3b bgr01 = t_orig.at<cv::Vec3b>( ( int ) l_orig_y, 1 + ( int ) l_orig_x );
            cv::Vec3b bgr10 = t_orig.at<cv::Vec3b>( 1 + ( int ) l_orig_y, ( int ) l_orig_x );
            cv::Vec3b bgr11 = t_orig.at<cv::Vec3b>( 1 + ( int ) l_orig_y, 1 + ( int ) l_orig_x );

            cv::Vec3b bgr;
            for ( int i = 0; i < 3; i++ )
            {
                // color calculation
                bgr[ i ] = bgr00[ i ] * ( 1 - l_diff_y ) * ( 1 - l_diff_x ) +
                           bgr01[ i ] * ( 1 - l_diff_y ) * ( l_diff_x ) +
                           bgr10[ i ] * ( l_diff_y ) * ( 1 - l_diff_x ) +
                           bgr11[ i ] * ( l_diff_y ) * ( l_diff_x );
                t_resize.at<cv::Vec3b>( l_resize_y, l_resize_x ) = bgr;
            }
        }

}

int main( int argn, char *args[] )
{
    if ( argn < 2 )
    {
        printf( "Use: %s rgb_picture\n", args[ 0 ] );
        exit( 1 );
    }
    cv::Mat l_orig = cv::imread( args[ 1 ], CV_LOAD_IMAGE_COLOR );
    cv::Mat l_resize_up( l_orig.rows * 2, l_orig.cols / 2 * 3, CV_8UC3 );
    cv::Mat l_resize_down( l_orig.rows / 2, l_orig.cols * 2 / 3, CV_8UC3 );
    cv::Mat l_blur0( l_orig.rows, l_orig.cols, CV_8UC3 );
    cv::Mat l_blur1( l_orig.rows, l_orig.cols, CV_8UC3 );
    int l_diagonal = sqrtf( l_orig.cols * l_orig.cols + l_orig.rows * l_orig.rows );
    cv::Mat l_rotate( l_diagonal, l_diagonal, CV_8UC3 );
    cv::Mat l_rotate_bad( l_diagonal, l_diagonal, CV_8UC3 );

    bilin_scale( l_orig, l_resize_up );
    bilin_scale( l_orig, l_resize_down );
    rotate_ok( l_orig, l_rotate, 3.14 / 4 );
    rotate_bad( l_orig, l_rotate_bad, -3.14 / 4 );
    blur( l_orig, l_blur0, 1.0 );
    for ( int i = 0; i < 10; i++ )
    {
        blur( l_blur0, l_blur1, 1.0 );
        blur( l_blur1, l_blur0, 1.0 );
    }

    cv::imshow( "l_orig", l_orig );
    cv::imshow( "l_resize_up", l_resize_up );
    cv::imshow( "l_resize_down", l_resize_down );
    cv::imshow( "l_rotate", l_rotate );
    cv::imshow( "l_rotate_bad", l_rotate_bad );
    cv::imshow( "l_blur", l_blur0 );

    cv::waitKey( 0 );

}


