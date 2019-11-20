#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <pthread.h>
#include <algorithm>    // std::swap
#include <iostream>
#include <cstring>

#define TYPE int

// class for one part of task
class TaskPart{
public:
    int m_id;                 // user thread identification 
    int m_from, m_to;     // data range
    TYPE *m_data;             // array
    TYPE *result;
    bool desc;

    TaskPart( int t_myid, int t_from, int t_length, TYPE *t_data, bool dsc ) :
        m_id( t_myid ), m_from( t_from ), m_to( t_length ), m_data( t_data ), desc(dsc) {
            this->result = new TYPE[m_to - m_from];
            std::memcpy(result, m_data + m_from, (m_to - m_from)*sizeof(TYPE));
            this->print_array();
        }

    void selection_sort(){
        int i, j, min_idx;  
        // One by one move boundary of unsorted subarray  
        for (i = 0; i < (m_to-m_from)-1; i++){  
            // Find the minimum/maximum element in unsorted array  
            min_idx = i;  
            for (j = i+1; j < (m_to-m_from); j++){
                if (this->desc){
                    if (result[j] > result[min_idx])  
                        min_idx = j;
                }
                else{
                    if (result[j] < result[min_idx])  
                        min_idx = j;
                }
            }
            std::swap(result[i], result[min_idx]);
        }
    }

        /* Function to print an array */
    void print_array(){  
        int i;  
        for (i=0; i < (m_to-m_from); i++)  
            std::cout << this->result[i] << " ";  
        std::cout << std::endl;  
    }
};

// Thread will search the largest element in array 
// from element arg->m_from with length of arg->m_length.
// Result will be stored to arg->m_max.
void *my_thread( void *t_void_arg )
{
    TaskPart *lp_task = ( TaskPart * ) t_void_arg;

    printf( "Thread %d started from %d with length %d...\n",
        lp_task->m_id, lp_task->m_from, lp_task->m_to );

    lp_task->selection_sort();

    printf( "Sorted array in thread %d: \n", lp_task->m_id);
    lp_task->print_array();

    return NULL;
}

// Time interval between two measurements converted to ms
int timeval_diff_to_ms( timeval *t_before, timeval *t_after )
{
    timeval l_res;
    timersub( t_after, t_before, &l_res );
    return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}

#define LENGTH_LIMIT 10


void mergeArrays(int arr1[], int arr2[], int n1, int n2, int arr3[]) { 
    int i = 0, j = 0, k = 0; 
    // Traverse both array 
    while (i<n1 && j <n2) { 
        // Check if current element of first 
        // array is smaller than current element 
        // of second array. If yes, store first 
        // array element and increment first array 
        // index. Otherwise do same with second array 
        if (arr1[i] < arr2[j]) 
            arr3[k++] = arr1[i++]; 
        else
            arr3[k++] = arr2[j++]; 
    } 
  
    // Store remaining elements of first array 
    while (i < n1) 
        arr3[k++] = arr1[i++]; 
  
    // Store remaining elements of second array 
    while (j < n2) 
        arr3[k++] = arr2[j++]; 
} 


int main( int t_na, char **t_arg ){
    // The number of elements must be used as program argument
    if ( t_na != 3 ) { 
        printf( "Specify number of elements, at least %d. And the order of sorting. ASC vs DESC \n", LENGTH_LIMIT ); 
        return 0; 
    }

    int l_my_length = atoi( t_arg[ 1 ] );
    char * order_sorting = t_arg[2];

    bool descending = (strcmp(order_sorting, "DESC") == 0); 

    if ( l_my_length < LENGTH_LIMIT ) { 
        printf( "The number of elements must be at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
    }

    // array allocation
    TYPE *l_my_array = new TYPE [ l_my_length ];

    if ( !l_my_array ) {
        printf( "Not enought memory for array!\n" );
        return 1;
    }

    // Initialization of random number generator
    srand( ( int ) time( NULL ) );

    printf( "Random numbers generetion started..." );
    for ( int i = 0; i < l_my_length; i++ ){
        l_my_array[ i ] = rand() % ( l_my_length * 10 );
        if ( !( i % LENGTH_LIMIT ) ) 
        {
            printf( "." ); 
            fflush( stdout );
        }
    }
    
    printf( "\nSelection sort using two threads...\n" );
    pthread_t l_pt1, l_pt2;
    TaskPart l_tp1( 1, 0, l_my_length / 2, l_my_array, descending );
    TaskPart l_tp2( 2, l_my_length / 2, l_my_length, l_my_array, descending );

    timeval l_time_before, l_time_after;

    // Time recording before searching
    gettimeofday( &l_time_before, NULL );

    // Threads starting
    pthread_create( &l_pt1, NULL, my_thread, &l_tp1 );
    pthread_create( &l_pt2, NULL, my_thread, &l_tp2 );

    // Waiting for threads completion 
    pthread_join( l_pt1, NULL );
    pthread_join( l_pt2, NULL );

    // Time recording after sorting
    gettimeofday( &l_time_after, NULL );

    //TODO: merge arrays
    TYPE * final_sorted = new TYPE[l_my_length];
    mergeArrays(l_tp1.result, l_tp2.result, l_my_length/2, l_my_length/2, final_sorted);
    std::cout << "Merged arrays: " << std::endl;
    for (int i=0; i < l_my_length; i++)  
        std::cout << final_sorted[i] << " ";  
    std::cout << std::endl; 
    printf( "The Sorting time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

    printf( "\nSelection sort using one thread...\n" );

    gettimeofday( &l_time_before, NULL );

    // Sorting in single thread
    TaskPart l_single( 333, 0, l_my_length, l_my_array, descending );
    l_single.selection_sort();

    gettimeofday( &l_time_after, NULL );

    printf( "Sorted array in thread %d: \n", l_single.m_id);
    l_single.print_array();

    printf( "The sorting time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
}
