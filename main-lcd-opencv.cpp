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
#include "font8x8.h"

#define LCD_WIDTH       320
#define LCD_HEIGHT      240
#define LCD_NAME        "Virtual LCD"

// LCD Simulator

// Virtual LCD
cv::Mat g_canvas( cv::Size( LCD_WIDTH, LCD_HEIGHT ), CV_8UC3 );

// Put color pixel on LCD (canvas)
void lcd_put_pixel( int t_x, int t_y, int t_rgb_565 )
{
    // Transform the color from a LCD form into the OpenCV form. 
    cv::Vec3b l_rgb_888( 
            (  t_rgb_565         & 0x1F ) << 3, 
            (( t_rgb_565 >> 5 )  & 0x3F ) << 2, 
            (( t_rgb_565 >> 11 ) & 0x1F ) << 3
            );
    g_canvas.at<cv::Vec3b>( t_y, t_x ) = l_rgb_888; // put pixel
}

// Clear LCD
void lcd_clear()
{
    cv::Vec3b l_black( 0, 0, 0 );
    g_canvas.setTo( l_black );
}

// LCD Initialization 
void lcd_init()
{
    cv::namedWindow( LCD_NAME, 0 );
    lcd_clear();
    cv::waitKey( 1 );
}

// Simple graphic interface

struct Point2D 
{
    int32_t x, y;
};

struct RGB
{
    uint8_t r, g, b;
};

class GraphElement
{
public:
    // foreground and background color
    RGB fg_color, bg_color;

    // constructor
    GraphElement( RGB t_fg_color, RGB t_bg_color ) : 
        fg_color( t_fg_color ), bg_color( t_bg_color ) {}

    // ONLY ONE INTERFACE WITH LCD HARDWARE!!!
    void drawPixel( int32_t t_x, int32_t t_y ) { lcd_put_pixel( t_x, t_y, convert_RGB888_to_RGB565( fg_color ) ); }
    
    // Draw graphics element
    virtual void draw() = 0;
    
    // Hide graphics element
    virtual void hide() { swap_fg_bg_color(); draw(); swap_fg_bg_color(); }
private:
    // swap foreground and backgroud colors
    void swap_fg_bg_color() { RGB l_tmp = fg_color; fg_color = bg_color; bg_color = l_tmp; } 

    // conversion of 24-bit RGB color into 16-bit color format
    int convert_RGB888_to_RGB565( RGB t_color ) { 
        uint8_t red   = t_color.r;
        uint8_t green = t_color.g;
        uint8_t blue  = t_color.b;
 
        uint16_t b = (blue >> 3) & 0x1f;
        uint16_t g = ((green >> 2) & 0x3f) << 5;
        uint16_t r = ((red >> 3) & 0x1f) << 11;
 
        return (uint16_t) (r | g | b);
    }
};


class Pixel : public GraphElement
{
public:
    // constructor
    Pixel( Point2D t_pos, RGB t_fg_color, RGB t_bg_color ) : pos( t_pos ), GraphElement( t_fg_color, t_bg_color ) {}
    // Draw method implementation
    virtual void draw() { drawPixel( pos.x, pos.y ); }
    // Position of Pixel
    Point2D pos;
};


class Circle : public GraphElement
{
public:
    // Center of circle
    Point2D center;
    // Radius of circle
    int32_t radius;

    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) : 
        center( t_center ), radius( t_radius ), GraphElement( t_fg, t_bg ) {};

    void draw() { 
        int f = 1 - this->radius;
        int ddF_x = 0;
        int ddF_y = -2 * this->radius;
        int x = 0;
        int y = this->radius;
    
        this->drawPixel(this->center.x, this->center.y + this->radius);
        this->drawPixel(this->center.x, this->center.y - this->radius);
        this->drawPixel(this->center.x + this->radius, this->center.y);
        this->drawPixel(this->center.x - this->radius, this->center.y);
    
        while(x < y){
            if(f >= 0){
                y--;
                ddF_y += 2;
                f += ddF_y;
            }
            x++;
            ddF_x += 2;
            f += ddF_x + 1;

           this->drawPixel(this->center.x + x, this->center.y + y);
           this->drawPixel(this->center.x - x, this->center.y + y);
           this->drawPixel(this->center.x + x, this->center.y - y);
           this->drawPixel(this->center.x - x, this->center.y - y);
           this->drawPixel(this->center.x + y, this->center.y + x);
           this->drawPixel(this->center.x - y, this->center.y + x);
           this->drawPixel(this->center.x + y, this->center.y - x);
           this->drawPixel(this->center.x - y, this->center.y - x);
        }
    }
};

class Character : public GraphElement 
{
public:
    // position of character
    Point2D pos;
    // character
    char character;

    Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg ) : 
      pos( t_pos ), character( t_char ), GraphElement( t_fg, t_bg ) {};

    void draw() {
        int WIDTH = 8;
        int HEIGHT = 8;
        for ( int y = 0; y < HEIGHT; y++ )
        {
            int radek_fontu = font8x8[ (int) this->character ][ y ];
            for ( int x = 0; x < WIDTH; x++ )
            {
                if ( radek_fontu & ( 1 << x ) ) this->drawPixel(x + this->pos.x, y + this->pos.y);
                //else this->hide();
            }
        }
     };
};

class Line : public GraphElement
{
public:
    // the first and the last point of line
    Point2D pos1, pos2;

    Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg ) : 
      pos1( t_pos1 ), pos2( t_pos2 ), GraphElement( t_fg, t_bg ) {}

    void draw() {  
        int x1 = this->pos1.x;
        int x2 = this->pos2.x;
        int y1 = this->pos1.y;
        int y2 = this->pos2.y;

        const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
        if(steep){
            std::swap(x1, y1);
            std::swap(x2, y2);
        }
        
        if(x1 > x2){
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        
        const float dx = x2 - x1;
        const float dy = fabs(y2 - y1);
        
        float error = dx / 2.0f;
        const int ystep = (y1 < y2) ? 1 : -1;
        int y = (int)y1;
        
        const int maxX = (int)x2;
        
        for(int x=(int)x1; x<maxX; x++)
        {
            if(steep){
                this->drawPixel(y,x);
            }
            else{
                this->drawPixel(x,y);
            }
        
            error -= dy;
            if(error < 0){
                y += ystep;
                error += dx;
            }
        }
    };
};

int main()
{
    lcd_init();                     // LCD initialization

    lcd_clear();                    // LCD clear screen
    Point2D point;
    point.x = 60;
    point.y = 60;

    Point2D point2;
    point2.x = 120;
    point2.y = 120;
    //140,45,25
    RGB fg_color;
    fg_color.r = 140;
    fg_color.g = 45;
    fg_color.b = 25;

    RGB bg_color;
    bg_color.r = 0;
    bg_color.g = 0;
    bg_color.b = 0;

    Character character = Character(point2, 'C', fg_color, bg_color);
    character.draw();

    // Line line = Line(point, point2, fg_color, bg_color);
    // line.draw();

    Circle circle = Circle(point2, 60, fg_color, bg_color);
    circle.draw();

    cv::imshow( LCD_NAME, g_canvas );   // refresh content of "LCD"
    cv::waitKey( 0 );                   // wait for key 
}