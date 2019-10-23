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
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "font8x8.h"

#define LCD_WIDTH       320
#define LCD_HEIGHT      240
#define LCD_NAME        "Virtual LCD"

float convert_to_radian(int angle){
    return M_PI*angle/180.0;
}

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

class Rectangle: public GraphElement{
public:
    Point2D p1,p2,p3,p4;
 
    Rectangle(Point2D pos1, Point2D pos2, Point2D pos3, Point2D pos4, RGB t_fg, RGB t_bg):
        p1(pos1), p2(pos2), p3(pos3), p4(pos4), GraphElement( t_fg, t_bg ) {}
 
    void draw(){
        Line l1 = Line(this->p1, this->p2, this->fg_color, this->bg_color);
        Line l2 = Line(this->p2, this->p3, this->fg_color, this->bg_color);
        Line l3 = Line(this->p3, this->p4, this->fg_color, this->bg_color);
        Line l4 = Line(this->p4, this->p1, this->fg_color, this->bg_color);
 
        l1.draw();
        l2.draw();
        l3.draw();
        l4.draw();
    }
};
 
class TimePanel: public GraphElement{
public:
    Point2D center;
    static const int RADIUS = 50;
    int current_pointer_angle;
    char identifier;
 
    TimePanel(Point2D c, RGB t_fg, RGB t_bg, char i):
        center(c),
        current_pointer_angle(0),
        identifier(i),
        GraphElement(t_fg, t_bg){
 
    }
 
    void draw(){

        this->drawPixel(this->center.x, this->center.y);
        this->drawPointer();
        this->drawClock();
        this->drawCapsule();
        this->drawIdentifier();
    }

    virtual void drawClock() = 0;

    virtual void drawPointer() = 0;

    void drawNumber(int num, Point2D p){
        if (num > 57){
            Character number = Character( p, (char) 49 , this->fg_color, this->bg_color );
            number.draw();
            
            Point2D p2;
            p2.x = p.x + 10;
            p2.y = p.y;

            Character number2 = Character( p2, (char) 47 + (num - 57) , this->fg_color, this->bg_color );
            number2.draw();
        }
        else{
            Character number = Character( p, (char) num , this->fg_color, this->bg_color );
            number.draw();
        }
    }

    void drawCapsule(){
        Point2D p1,p2,p3,p4;

        p1.x = this->center.x - RADIUS - 10;
        p1.y = this->center.y - RADIUS - 10;

        p2.x = this->center.x + RADIUS + 20;
        p2.y = p1.y;

        p3.x = p2.x;
        p3.y = this->center.y + RADIUS/2;

        p4.x = p1.x;
        p4.y = p3.y;

        Rectangle rectangle = Rectangle(p1, p2, p3, p4, this->fg_color, this->bg_color);
        rectangle.draw();
    }

    void drawIdentifier(){
        Point2D p;

        p.x = this->center.x;
        p.y = this->center.y - RADIUS/2;

        Character ch = Character(p, this->identifier, this->fg_color, this->bg_color);
        ch.draw();
    }
};

class HoursPanel : public TimePanel{
public:
    int division_size = 12;
    HoursPanel(Point2D c, RGB t_fg, RGB t_bg): TimePanel(c, t_fg, t_bg, 72){}

    void drawClock(){
        int angle_difference = 180/(this->division_size - 1);
        int current_angle = 0;
        int number = 49;

        while(current_angle <= 180){
            int px = this->center.x - RADIUS * cos(convert_to_radian(current_angle));
            int py = this->center.y - RADIUS * sin(convert_to_radian(current_angle));
            Point2D p;
            p.x = px;
            p.y = py;
            this->drawNumber(number, p);
            current_angle += angle_difference;
            number += 1;
        }
    }

    void drawPointer(){
        Point2D original;

        int modifier_x = cos(convert_to_radian(this->current_pointer_angle))/abs(cos(convert_to_radian(this->current_pointer_angle)));
        
        original.x = this->center.x - RADIUS * cos(convert_to_radian(this->current_pointer_angle)) + 15*modifier_x;
        original.y = this->center.y - RADIUS * sin(convert_to_radian(this->current_pointer_angle)) + 3;
 
        Line l = Line(this->center, original, this->fg_color, this->bg_color);

        l.draw(); 
    }

    void tick(int d_size){
        int increase = 180/d_size;
        this->current_pointer_angle += increase;
    }
};

class MinutesSecondsPanel : public TimePanel{
public:
    int division_size = 60;
    MinutesSecondsPanel(Point2D c, RGB t_fg, RGB t_bg, char id): TimePanel(c, t_fg, t_bg, id){}

    void drawClock(){
        int angle_difference = 180/(this->division_size - 1);
        int current_angle = 0;
        int number = 49;

        while(current_angle <= 180){
            int px = this->center.x - RADIUS * cos(convert_to_radian(current_angle));
            int py = this->center.y - RADIUS * sin(convert_to_radian(current_angle));
            Point2D p;
            p.x = px;
            p.y = py;
            if(current_angle == 90){
                this->drawPixel(px, py);
            }
            else{
                this->drawPixel(px, py);
            }
            
            current_angle += angle_difference;
            number += 1;
        }
    }

    void drawPointer(){
        Point2D original;

        int modifier_x = cos(convert_to_radian(this->current_pointer_angle))/abs(cos(convert_to_radian(this->current_pointer_angle)));
        
        original.x = this->center.x - RADIUS * cos(convert_to_radian(this->current_pointer_angle)) + 15*modifier_x;
        original.y = this->center.y - RADIUS * sin(convert_to_radian(this->current_pointer_angle));
 
        Line l = Line(this->center, original, this->fg_color, this->bg_color);

        l.draw(); 
    }
};

class Clocks{
public:
    HoursPanel hours;
    MinutesSecondsPanel minutes;
    MinutesSecondsPanel seconds;

    Clocks(HoursPanel h, MinutesSecondsPanel m, MinutesSecondsPanel s) : hours(h), minutes(m), seconds(s){}

    void draw(){
        this->hours.draw();
        this->minutes.draw();
        this->seconds.draw();
    } 
};

int main()
{
    lcd_init();                     // LCD initialization

    lcd_clear();                    // LCD clear screen
    
    RGB color1, color2;
    Point2D p1,p2,p3;
 
    color1.r = 255;
    color1.g = 0;
    color1.b = 0;
 
    color2.r = 255;
    color2.g = 255;
    color2.b = 255;
 
    p1.x = 80;
    p1.y = 80;

    p2.x = 240;
    p2.y = 80;

    p3.x = 140;
    p3.y = 175;

    HoursPanel hours = HoursPanel(p1, color1, color2);
    hours.draw();

    MinutesSecondsPanel minutes = MinutesSecondsPanel(p2, color1, color2, 77);
    minutes.draw();

    MinutesSecondsPanel seconds = MinutesSecondsPanel(p3, color1, color2, 83);
    seconds.draw();


    cv::imshow( LCD_NAME, g_canvas );   // refresh content of "LCD"
    cv::waitKey( 0 );                   // wait for key 
}