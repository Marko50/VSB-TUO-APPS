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

    TaskPart(){

    }

    TaskPart( int t_myid, int t_from, int t_length, TYPE *t_data, bool dsc ) :
        m_id( t_myid ), m_from( t_from ), m_to( t_length ), m_data( t_data ), desc(dsc) {
            this->result = new TYPE[m_to - m_from];
            std::memcpy(result, m_data + m_from, (m_to - m_from)*sizeof(TYPE));
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

    printf( "\nThread %d started from %d to %d...\n",
        lp_task->m_id, lp_task->m_from, lp_task->m_to );

    lp_task->selection_sort();

    // printf( "\nSorted array in thread %d: \n", lp_task->m_id);
    // lp_task->print_array();

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


void mergeArrays(TYPE arr1[], TYPE arr2[], int n1, int n2, TYPE arr3[], bool desc) { 
    int i = 0, j = 0, k = 0; 
    // Traverse both array 
    while (i<n1 && j <n2) { 
        // Check if current element of first 
        // array is smaller than current element 
        // of second array. If yes, store first 
        // array element and increment first array 
        // index. Otherwise do same with second array 
        if(desc){
            if (arr1[i] > arr2[j]) 
                arr3[k++] = arr1[i++]; 
            else
                arr3[k++] = arr2[j++]; 
        }
        else{
            if (arr1[i] < arr2[j]) 
                arr3[k++] = arr1[i++]; 
            else
                arr3[k++] = arr2[j++]; 
        }  
    } 
    // Store remaining elements of first array 
    while (i < n1) 
        arr3[k++] = arr1[i++]; 
  
    // Store remaining elements of second array 
    while (j < n2) 
        arr3[k++] = arr2[j++];
} 

bool compare_arrays(TYPE * arr1, TYPE * arr2, int n){
    for (int i = 0; i < n; i++){
        if (arr1[i] != arr2[i]){
            return false;
        }
    }
    return true;
}

void sort_array_using_threads(int number_of_threads, TYPE * arr, int length, bool desc){
    printf( "\nSelection sort using %d threads...\n", number_of_threads);
    
    pthread_t threads[number_of_threads];
    TaskPart tasks[number_of_threads];

    for (int i = 0; i < number_of_threads; i++){
        TaskPart l_tp( i + 1, (length / number_of_threads) * i, (length / number_of_threads)*i + (length / number_of_threads), arr, desc );
        tasks[i] = l_tp;
    }

    for (int i = 0; i < number_of_threads; i++){
        pthread_create(&threads[i], NULL, my_thread, &tasks[i]);
    }

    for (int i = 0; i < number_of_threads; i++){
        pthread_join(threads[i], NULL);
    }

    TYPE * accum = new TYPE[length/number_of_threads];
    memcpy(accum, tasks[0].result, sizeof(TYPE) * (length/number_of_threads));
    for (int i = 1; i < number_of_threads; i++){
        TYPE * copy = new TYPE[ (length/number_of_threads) * i];
        memcpy(copy, accum, sizeof(TYPE) * (length/number_of_threads) * i);
        mergeArrays(copy, tasks[i].result, (length/number_of_threads) * i, length/number_of_threads, accum, desc);
    }
    memcpy(arr, accum, sizeof(TYPE) * length);

    // printf("\nFinal sorted:\n");
    //     for(int i = 0; i < length; i++){
    //         printf("%d ", arr[i]);
    //     }
    // printf("\n");
}

int main( int t_na, char **t_arg ){
    // The number of elements must be used as program argument
    if ( t_na != 2 ) { 
        printf( "Specify number of elements, at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
    }

    int l_my_length = atoi( t_arg[ 1 ] );

    if ( l_my_length < LENGTH_LIMIT ) { 
        printf( "The number of elements must be at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
    }

    // array allocation
    TYPE *l_my_array = new TYPE [ l_my_length ];
    TYPE *l_my_array_copy = new TYPE [l_my_length];

    if ( !l_my_array ) {
        printf( "Not enought memory for array!\n" );
        return 1;
    }

    // Initialization of random number generator
    srand( ( int ) time( NULL ) );

    printf( "Random numbers generetion started..." );
    for ( int i = 0; i < l_my_length; i++ ){
        l_my_array[ i ] = rand() % ( 100000 );
        if ( !( i % LENGTH_LIMIT ) ) {
            printf( "." ); 
            fflush( stdout );
        }
    }

    memcpy(l_my_array_copy, l_my_array, l_my_length*sizeof(TYPE));
    
    timeval l_time_before, l_time_after;

    printf("\nNow sorting asc..\n");
    gettimeofday( &l_time_before, NULL );
    sort_array_using_threads(4, l_my_array, l_my_length, false);
    gettimeofday( &l_time_after, NULL );
    printf( "The Sorting time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
     
    printf("\nNow sorting the sorted asc..\n");
    gettimeofday( &l_time_before, NULL );
    sort_array_using_threads(4, l_my_array, l_my_length, false);
    gettimeofday( &l_time_after, NULL );
    printf( "The Sorting time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

    // Sorting in single thread
    gettimeofday( &l_time_before, NULL );
    sort_array_using_threads(1, l_my_array_copy, l_my_length, false);
    gettimeofday( &l_time_after, NULL );
    printf( "The sorting time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );

    if (compare_arrays(l_my_array, l_my_array_copy, l_my_length)){
        printf("\nArrays are the same and correctly sorted.\n");
    }
    
    printf("\nNow sorting the asc sorted desc..\n");
    gettimeofday( &l_time_before, NULL );
    sort_array_using_threads(4, l_my_array, l_my_length, true);
    gettimeofday( &l_time_after, NULL );
    printf( "The Sorting time: %d [ms]\n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
}
