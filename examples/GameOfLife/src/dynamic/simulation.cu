
  /*
  * FLAME GPU v 1.5.X for CUDA 9
  * Copyright University of Sheffield.
  * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
  * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
  *
  * University of Sheffield retain all intellectual property and
  * proprietary rights in and to this software and related documentation.
  * Any use, reproduction, disclosure, or distribution of this software
  * and related documentation without an express license agreement from
  * University of Sheffield is strictly prohibited.
  *
  * For terms of licence agreement please attached licence or view licence
  * on www.flamegpu.com website.
  *
  */

  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */

/* cell Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_cell_list* d_cells;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_cell_list* d_cells_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_cell_list* d_cells_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_cell_count;   /**< Agent population size counter */ 
int h_xmachine_memory_cell_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_cell_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_cell_values;  /**< Agent sort identifiers value */

/* cell state variables */
xmachine_memory_cell_list* h_cells_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_cell_list* d_cells_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_cell_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_cells_default_variable_state_data_iteration;
unsigned int h_cells_default_variable_x_data_iteration;
unsigned int h_cells_default_variable_y_data_iteration;


/* Message Memory */

/* state Message variables */
xmachine_message_state_list* h_states;         /**< Pointer to message list on host*/
xmachine_message_state_list* d_states;         /**< Pointer to message list on device*/
xmachine_message_state_list* d_states_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_state_range;     /**< range of the discrete message*/
int h_message_state_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_state_state_offset;
int h_tex_xmachine_message_state_x_offset;
int h_tex_xmachine_message_state_y_offset;
  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_cell;
size_t temp_scan_storage_bytes_cell;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** cell_output_state
 * Agent function prototype for output_state function of cell agent
 */
void cell_output_state(cudaStream_t &stream);

/** cell_update_state
 * Agent function prototype for update_state function of cell agent
 */
void cell_update_state(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
    // Initialise some global variables
    g_iterationNumber = 0;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_cells_default_variable_state_data_iteration = 0;
    h_cells_default_variable_x_data_iteration = 0;
    h_cells_default_variable_y_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_cell_SoA_size = sizeof(xmachine_memory_cell_list);
	h_cells_default = (xmachine_memory_cell_list*)malloc(xmachine_cell_SoA_size);

	/* Message memory allocation (CPU) */
	int message_state_SoA_size = sizeof(xmachine_message_state_list);
	h_states = (xmachine_message_state_list*)malloc(message_state_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs
	
	/* Set discrete state message variables (range, width)*/
	h_message_state_range = 1; //from xml
	h_message_state_width = (int)floor(sqrt((float)xmachine_message_state_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_state_MAX)){
		printf("ERROR: state message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_state_range, &h_message_state_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_state_width, &h_message_state_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_cell_MAX)){
		printf("ERROR: cells agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_cell_pop_width = (int)sqrt(xmachine_memory_cell_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_cells_default, &h_xmachine_memory_cell_default_count);
	
	
	/* cell Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_cells, xmachine_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_cells_swap, xmachine_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_cells_new, xmachine_cell_SoA_size));
    
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_cells_default, xmachine_cell_SoA_size));
	gpuErrchk( cudaMemcpy( d_cells_default, h_cells_default, xmachine_cell_SoA_size, cudaMemcpyHostToDevice));
    
	/* state Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_states, message_state_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_states_swap, message_state_SoA_size));
	gpuErrchk( cudaMemcpy( d_states, h_states, message_state_SoA_size, cudaMemcpyHostToDevice));
		

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_cell = nullptr;
    temp_scan_storage_bytes_cell = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_cell, 
        temp_scan_storage_bytes_cell, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_cell_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_cell, temp_scan_storage_bytes_cell));
    

	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_cell_default_count: %u\n",get_agent_cell_default_count());
	
#endif
} 



void cleanup(){

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* cell Agent variables */
	gpuErrchk(cudaFree(d_cells));
	gpuErrchk(cudaFree(d_cells_swap));
	gpuErrchk(cudaFree(d_cells_new));
	
	free( h_cells_default);
	gpuErrchk(cudaFree(d_cells_default));
	

	/* Message data free */
	
	/* state Message variables */
	free( h_states);
	gpuErrchk(cudaFree(d_states));
	gpuErrchk(cudaFree(d_states_swap));
	

    /* Free temporary CUB memory */
    
    gpuErrchk(cudaFree(d_temp_scan_storage_cell));
    d_temp_scan_storage_cell = nullptr;
    temp_scan_storage_bytes_cell = 0;
    
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

	/* set all non partitioned and spatial partitioned message counts to 0*/

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	cell_output_state(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: cell_output_state = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	cell_update_state(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: cell_update_state = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_cell_default_count: %u\n",get_agent_cell_default_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* Environment functions */

//host constant declaration



/* Agent data access functions*/

    
int get_agent_cell_MAX_count(){
    return xmachine_memory_cell_MAX;
}


int get_agent_cell_default_count(){
	//discrete agent 
	return xmachine_memory_cell_MAX;
}

xmachine_memory_cell_list* get_device_cell_default_agents(){
	return d_cells_default;
}

xmachine_memory_cell_list* get_host_cell_default_agents(){
	return h_cells_default;
}

int get_cell_population_width(){
  return h_xmachine_memory_cell_pop_width;
}



/* Host based access of agent variables*/

/** int get_cell_default_variable_state(unsigned int index)
 * Gets the value of the state variable of an cell agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable state
 */
__host__ int get_cell_default_variable_state(unsigned int index){
    unsigned int count = get_agent_cell_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_cells_default_variable_state_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_cells_default->state,
                    d_cells_default->state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_cells_default_variable_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_cells_default->state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access state for the %u th member of cell_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_cell_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an cell agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_cell_default_variable_x(unsigned int index){
    unsigned int count = get_agent_cell_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_cells_default_variable_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_cells_default->x,
                    d_cells_default->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_cells_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_cells_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of cell_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_cell_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an cell agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_cell_default_variable_y(unsigned int index){
    unsigned int count = get_agent_cell_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_cells_default_variable_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_cells_default->y,
                    d_cells_default->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_cells_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_cells_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of cell_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/*  Analytics Functions */

int reduce_cell_default_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_cells_default->state),  thrust::device_pointer_cast(d_cells_default->state) + h_xmachine_memory_cell_default_count);
}
int count_cell_default_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_cells_default->state),  thrust::device_pointer_cast(d_cells_default->state) + h_xmachine_memory_cell_default_count, count_value);
}
int reduce_cell_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_cells_default->x),  thrust::device_pointer_cast(d_cells_default->x) + h_xmachine_memory_cell_default_count);
}
int count_cell_default_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_cells_default->x),  thrust::device_pointer_cast(d_cells_default->x) + h_xmachine_memory_cell_default_count, count_value);
}
int reduce_cell_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_cells_default->y),  thrust::device_pointer_cast(d_cells_default->y) + h_xmachine_memory_cell_default_count);
}
int count_cell_default_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_cells_default->y),  thrust::device_pointer_cast(d_cells_default->y) + h_xmachine_memory_cell_default_count, count_value);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int cell_output_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** cell_output_state
 * Agent function prototype for output_state function of cell agent
 */
void cell_output_state(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_cell_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_cell_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_cell_list* cells_default_temp = d_cells;
	d_cells = d_cells_default;
	d_cells_default = cells_default_temp;
	//set working count to current state count
	h_xmachine_memory_cell_count = h_xmachine_memory_cell_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_count, &h_xmachine_memory_cell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_cell_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_state, cell_output_state_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = cell_output_state_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_state)
	//Reallocate   : false
	//Input        : 
	//Output       : state
	//Agent Output : 
	GPUFLAME_output_state<<<g, b, sm_size, stream>>>(d_cells, d_states);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	cells_default_temp = d_cells_default;
	d_cells_default = d_cells;
	d_cells = cells_default_temp;
    //set current state count
	h_xmachine_memory_cell_default_count = h_xmachine_memory_cell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int cell_update_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_state_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_state)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** cell_update_state
 * Agent function prototype for update_state function of cell agent
 */
void cell_update_state(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_cell_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_cell_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_cell_list* cells_default_temp = d_cells;
	d_cells = d_cells_default;
	d_cells_default = cells_default_temp;
	//set working count to current state count
	h_xmachine_memory_cell_count = h_xmachine_memory_cell_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_count, &h_xmachine_memory_cell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_cell_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_update_state, cell_update_state_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = cell_update_state_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_state_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_state_state_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_state_state_byte_offset, tex_xmachine_message_state_state, d_states->state, sizeof(int)*xmachine_message_state_MAX));
	h_tex_xmachine_message_state_state_offset = (int)tex_xmachine_message_state_state_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_state_state_offset, &h_tex_xmachine_message_state_state_offset, sizeof(int)));
	size_t tex_xmachine_message_state_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_state_x_byte_offset, tex_xmachine_message_state_x, d_states->x, sizeof(int)*xmachine_message_state_MAX));
	h_tex_xmachine_message_state_x_offset = (int)tex_xmachine_message_state_x_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_state_x_offset, &h_tex_xmachine_message_state_x_offset, sizeof(int)));
	size_t tex_xmachine_message_state_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_state_y_byte_offset, tex_xmachine_message_state_y, d_states->y, sizeof(int)*xmachine_message_state_MAX));
	h_tex_xmachine_message_state_y_offset = (int)tex_xmachine_message_state_y_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_state_y_offset, &h_tex_xmachine_message_state_y_offset, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (update_state)
	//Reallocate   : false
	//Input        : state
	//Output       : 
	//Agent Output : 
	GPUFLAME_update_state<<<g, b, sm_size, stream>>>(d_cells, d_states);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_state_state));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_state_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_state_y));
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	cells_default_temp = d_cells_default;
	d_cells_default = d_cells;
	d_cells = cells_default_temp;
    //set current state count
	h_xmachine_memory_cell_default_count = h_xmachine_memory_cell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_cell_default_count, &h_xmachine_memory_cell_default_count, sizeof(int)));	
	
	
}


 
extern void reset_cell_default_count()
{
    h_xmachine_memory_cell_default_count = 0;
}
