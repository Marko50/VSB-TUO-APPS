#include <stdio.h>
#include "font8x8.h"

#define WIDTH 8
#define HEIGHT 8

int main()
{
    for ( int znak = 0; znak < 256; znak++ ) // cela ASCII tabulka
    {
        for ( int y = 0; y < HEIGHT; y++ )
        {
            int radek_fontu = font8x8[ znak ][ y ];
            for ( int x = 0; x < WIDTH; x++ )
            {
                if ( radek_fontu & ( 1 << x ) ) printf( "#" );
                else printf( " " );
            }
            printf( "\n" );
        }
    }
}
