
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
#include <thrust/extrema.h>
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

/* FloodCell Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_FloodCell_list* d_FloodCells;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_FloodCell_list* d_FloodCells_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_FloodCell_list* d_FloodCells_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_FloodCell_count;   /**< Agent population size counter */ 
int h_xmachine_memory_FloodCell_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_FloodCell_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_FloodCell_values;  /**< Agent sort identifiers value */

/* FloodCell state variables */
xmachine_memory_FloodCell_list* h_FloodCells_Default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_FloodCell_list* d_FloodCells_Default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_FloodCell_Default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_FloodCells_Default_variable_inDomain_data_iteration;
unsigned int h_FloodCells_Default_variable_x_data_iteration;
unsigned int h_FloodCells_Default_variable_y_data_iteration;
unsigned int h_FloodCells_Default_variable_z0_data_iteration;
unsigned int h_FloodCells_Default_variable_h_data_iteration;
unsigned int h_FloodCells_Default_variable_qx_data_iteration;
unsigned int h_FloodCells_Default_variable_qy_data_iteration;
unsigned int h_FloodCells_Default_variable_timeStep_data_iteration;
unsigned int h_FloodCells_Default_variable_minh_loc_data_iteration;
unsigned int h_FloodCells_Default_variable_hFace_E_data_iteration;
unsigned int h_FloodCells_Default_variable_etFace_E_data_iteration;
unsigned int h_FloodCells_Default_variable_qxFace_E_data_iteration;
unsigned int h_FloodCells_Default_variable_qyFace_E_data_iteration;
unsigned int h_FloodCells_Default_variable_hFace_W_data_iteration;
unsigned int h_FloodCells_Default_variable_etFace_W_data_iteration;
unsigned int h_FloodCells_Default_variable_qxFace_W_data_iteration;
unsigned int h_FloodCells_Default_variable_qyFace_W_data_iteration;
unsigned int h_FloodCells_Default_variable_hFace_N_data_iteration;
unsigned int h_FloodCells_Default_variable_etFace_N_data_iteration;
unsigned int h_FloodCells_Default_variable_qxFace_N_data_iteration;
unsigned int h_FloodCells_Default_variable_qyFace_N_data_iteration;
unsigned int h_FloodCells_Default_variable_hFace_S_data_iteration;
unsigned int h_FloodCells_Default_variable_etFace_S_data_iteration;
unsigned int h_FloodCells_Default_variable_qxFace_S_data_iteration;
unsigned int h_FloodCells_Default_variable_qyFace_S_data_iteration;


/* Message Memory */

/* WetDryMessage Message variables */
xmachine_message_WetDryMessage_list* h_WetDryMessages;         /**< Pointer to message list on host*/
xmachine_message_WetDryMessage_list* d_WetDryMessages;         /**< Pointer to message list on device*/
xmachine_message_WetDryMessage_list* d_WetDryMessages_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_WetDryMessage_range;     /**< range of the discrete message*/
int h_message_WetDryMessage_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_WetDryMessage_inDomain_offset;
int h_tex_xmachine_message_WetDryMessage_x_offset;
int h_tex_xmachine_message_WetDryMessage_y_offset;
int h_tex_xmachine_message_WetDryMessage_min_hloc_offset;
/* SpaceOperatorMessage Message variables */
xmachine_message_SpaceOperatorMessage_list* h_SpaceOperatorMessages;         /**< Pointer to message list on host*/
xmachine_message_SpaceOperatorMessage_list* d_SpaceOperatorMessages;         /**< Pointer to message list on device*/
xmachine_message_SpaceOperatorMessage_list* d_SpaceOperatorMessages_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_SpaceOperatorMessage_range;     /**< range of the discrete message*/
int h_message_SpaceOperatorMessage_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_x_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_y_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset;
int h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset;
  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_FloodCell;
size_t temp_scan_storage_bytes_FloodCell;


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

/** FloodCell_PrepareWetDry
 * Agent function prototype for PrepareWetDry function of FloodCell agent
 */
void FloodCell_PrepareWetDry(cudaStream_t &stream);

/** FloodCell_ProcessWetDryMessage
 * Agent function prototype for ProcessWetDryMessage function of FloodCell agent
 */
void FloodCell_ProcessWetDryMessage(cudaStream_t &stream);

/** FloodCell_PrepareSpaceOperator
 * Agent function prototype for PrepareSpaceOperator function of FloodCell agent
 */
void FloodCell_PrepareSpaceOperator(cudaStream_t &stream);

/** FloodCell_ProcessSpaceOperatorMessage
 * Agent function prototype for ProcessSpaceOperatorMessage function of FloodCell agent
 */
void FloodCell_ProcessSpaceOperatorMessage(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
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
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
    // Initialise some global variables
    g_iterationNumber = 0;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_FloodCells_Default_variable_inDomain_data_iteration = 0;
    h_FloodCells_Default_variable_x_data_iteration = 0;
    h_FloodCells_Default_variable_y_data_iteration = 0;
    h_FloodCells_Default_variable_z0_data_iteration = 0;
    h_FloodCells_Default_variable_h_data_iteration = 0;
    h_FloodCells_Default_variable_qx_data_iteration = 0;
    h_FloodCells_Default_variable_qy_data_iteration = 0;
    h_FloodCells_Default_variable_timeStep_data_iteration = 0;
    h_FloodCells_Default_variable_minh_loc_data_iteration = 0;
    h_FloodCells_Default_variable_hFace_E_data_iteration = 0;
    h_FloodCells_Default_variable_etFace_E_data_iteration = 0;
    h_FloodCells_Default_variable_qxFace_E_data_iteration = 0;
    h_FloodCells_Default_variable_qyFace_E_data_iteration = 0;
    h_FloodCells_Default_variable_hFace_W_data_iteration = 0;
    h_FloodCells_Default_variable_etFace_W_data_iteration = 0;
    h_FloodCells_Default_variable_qxFace_W_data_iteration = 0;
    h_FloodCells_Default_variable_qyFace_W_data_iteration = 0;
    h_FloodCells_Default_variable_hFace_N_data_iteration = 0;
    h_FloodCells_Default_variable_etFace_N_data_iteration = 0;
    h_FloodCells_Default_variable_qxFace_N_data_iteration = 0;
    h_FloodCells_Default_variable_qyFace_N_data_iteration = 0;
    h_FloodCells_Default_variable_hFace_S_data_iteration = 0;
    h_FloodCells_Default_variable_etFace_S_data_iteration = 0;
    h_FloodCells_Default_variable_qxFace_S_data_iteration = 0;
    h_FloodCells_Default_variable_qyFace_S_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_FloodCell_SoA_size = sizeof(xmachine_memory_FloodCell_list);
	h_FloodCells_Default = (xmachine_memory_FloodCell_list*)malloc(xmachine_FloodCell_SoA_size);

	/* Message memory allocation (CPU) */
	int message_WetDryMessage_SoA_size = sizeof(xmachine_message_WetDryMessage_list);
	h_WetDryMessages = (xmachine_message_WetDryMessage_list*)malloc(message_WetDryMessage_SoA_size);
	int message_SpaceOperatorMessage_SoA_size = sizeof(xmachine_message_SpaceOperatorMessage_list);
	h_SpaceOperatorMessages = (xmachine_message_SpaceOperatorMessage_list*)malloc(message_SpaceOperatorMessage_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs
    PROFILE_POP_RANGE(); //"allocate host"
	
	
	/* Set discrete WetDryMessage message variables (range, width)*/
	h_message_WetDryMessage_range = 1; //from xml
	h_message_WetDryMessage_width = (int)floor(sqrt((float)xmachine_message_WetDryMessage_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_WetDryMessage_MAX)){
		printf("ERROR: WetDryMessage message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_WetDryMessage_range, &h_message_WetDryMessage_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_WetDryMessage_width, &h_message_WetDryMessage_width, sizeof(int)));
	
	
	/* Set discrete SpaceOperatorMessage message variables (range, width)*/
	h_message_SpaceOperatorMessage_range = 1; //from xml
	h_message_SpaceOperatorMessage_width = (int)floor(sqrt((float)xmachine_message_SpaceOperatorMessage_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_SpaceOperatorMessage_MAX)){
		printf("ERROR: SpaceOperatorMessage message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_SpaceOperatorMessage_range, &h_message_SpaceOperatorMessage_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_SpaceOperatorMessage_width, &h_message_SpaceOperatorMessage_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_FloodCell_MAX)){
		printf("ERROR: FloodCells agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_FloodCell_pop_width = (int)sqrt(xmachine_memory_FloodCell_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_FloodCells_Default, &h_xmachine_memory_FloodCell_Default_count);
	

    PROFILE_PUSH_RANGE("allocate device");
	
	/* FloodCell Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells, xmachine_FloodCell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells_swap, xmachine_FloodCell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells_new, xmachine_FloodCell_SoA_size));
    
	/* Default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells_Default, xmachine_FloodCell_SoA_size));
	gpuErrchk( cudaMemcpy( d_FloodCells_Default, h_FloodCells_Default, xmachine_FloodCell_SoA_size, cudaMemcpyHostToDevice));
    
	/* WetDryMessage Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_WetDryMessages, message_WetDryMessage_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_WetDryMessages_swap, message_WetDryMessage_SoA_size));
	gpuErrchk( cudaMemcpy( d_WetDryMessages, h_WetDryMessages, message_WetDryMessage_SoA_size, cudaMemcpyHostToDevice));
	
	/* SpaceOperatorMessage Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_SpaceOperatorMessages, message_SpaceOperatorMessage_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_SpaceOperatorMessages_swap, message_SpaceOperatorMessage_SoA_size));
	gpuErrchk( cudaMemcpy( d_SpaceOperatorMessages, h_SpaceOperatorMessages, message_SpaceOperatorMessage_SoA_size, cudaMemcpyHostToDevice));
		
    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_FloodCell = nullptr;
    temp_scan_storage_bytes_FloodCell = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_FloodCell, 
        temp_scan_storage_bytes_FloodCell, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_FloodCell_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_FloodCell, temp_scan_storage_bytes_FloodCell));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
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

    PROFILE_POP_RANGE();

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

	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    initConstants();
    PROFILE_PUSH_RANGE("initConstants");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: initConstants = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_FloodCell_Default_count: %u\n",get_agent_FloodCell_Default_count());
	
#endif
} 



void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* FloodCell Agent variables */
	gpuErrchk(cudaFree(d_FloodCells));
	gpuErrchk(cudaFree(d_FloodCells_swap));
	gpuErrchk(cudaFree(d_FloodCells_new));
	
	free( h_FloodCells_Default);
	gpuErrchk(cudaFree(d_FloodCells_Default));
	

	/* Message data free */
	
	/* WetDryMessage Message variables */
	free( h_WetDryMessages);
	gpuErrchk(cudaFree(d_WetDryMessages));
	gpuErrchk(cudaFree(d_WetDryMessages_swap));
	
	/* SpaceOperatorMessage Message variables */
	free( h_SpaceOperatorMessages);
	gpuErrchk(cudaFree(d_SpaceOperatorMessages));
	gpuErrchk(cudaFree(d_SpaceOperatorMessages_swap));
	

    /* Free temporary CUB memory */
    
    gpuErrchk(cudaFree(d_temp_scan_storage_FloodCell));
    d_temp_scan_storage_FloodCell = nullptr;
    temp_scan_storage_bytes_FloodCell = 0;
    
  
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
PROFILE_SCOPED_RANGE("singleIteration");

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
	
    PROFILE_PUSH_RANGE("FloodCell_PrepareWetDry");
	FloodCell_PrepareWetDry(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: FloodCell_PrepareWetDry = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("FloodCell_ProcessWetDryMessage");
	FloodCell_ProcessWetDryMessage(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: FloodCell_ProcessWetDryMessage = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("FloodCell_PrepareSpaceOperator");
	FloodCell_PrepareSpaceOperator(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: FloodCell_PrepareSpaceOperator = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("FloodCell_ProcessSpaceOperatorMessage");
	FloodCell_ProcessSpaceOperatorMessage(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: FloodCell_ProcessSpaceOperatorMessage = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("DELTA_T_func");
	DELTA_T_func();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: DELTA_T_func = %f (ms)\n", instrument_milliseconds);
#endif

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_FloodCell_Default_count: %u\n",get_agent_FloodCell_Default_count());
	
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
double h_env_dt;


//constant setter
void set_dt(double* h_dt){
    gpuErrchk(cudaMemcpyToSymbol(dt, h_dt, sizeof(double)));
    memcpy(&h_env_dt, h_dt,sizeof(double));
}

//constant getter
const double* get_dt(){
    return &h_env_dt;
}




/* Agent data access functions*/

    
int get_agent_FloodCell_MAX_count(){
    return xmachine_memory_FloodCell_MAX;
}


int get_agent_FloodCell_Default_count(){
	//discrete agent 
	return xmachine_memory_FloodCell_MAX;
}

xmachine_memory_FloodCell_list* get_device_FloodCell_Default_agents(){
	return d_FloodCells_Default;
}

xmachine_memory_FloodCell_list* get_host_FloodCell_Default_agents(){
	return h_FloodCells_Default;
}

int get_FloodCell_population_width(){
  return h_xmachine_memory_FloodCell_pop_width;
}



/* Host based access of agent variables*/

/** int get_FloodCell_Default_variable_inDomain(unsigned int index)
 * Gets the value of the inDomain variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable inDomain
 */
__host__ int get_FloodCell_Default_variable_inDomain(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_inDomain_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->inDomain,
                    d_FloodCells_Default->inDomain,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_inDomain_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->inDomain[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access inDomain for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_FloodCell_Default_variable_x(unsigned int index)
 * Gets the value of the x variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_FloodCell_Default_variable_x(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->x,
                    d_FloodCells_Default->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_FloodCell_Default_variable_y(unsigned int index)
 * Gets the value of the y variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_FloodCell_Default_variable_y(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->y,
                    d_FloodCells_Default->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_z0(unsigned int index)
 * Gets the value of the z0 variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z0
 */
__host__ double get_FloodCell_Default_variable_z0(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_z0_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->z0,
                    d_FloodCells_Default->z0,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_z0_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->z0[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z0 for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_h(unsigned int index)
 * Gets the value of the h variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable h
 */
__host__ double get_FloodCell_Default_variable_h(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_h_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->h,
                    d_FloodCells_Default->h,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_h_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->h[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access h for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qx(unsigned int index)
 * Gets the value of the qx variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qx
 */
__host__ double get_FloodCell_Default_variable_qx(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qx_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qx,
                    d_FloodCells_Default->qx,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qx for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qy(unsigned int index)
 * Gets the value of the qy variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qy
 */
__host__ double get_FloodCell_Default_variable_qy(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qy_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qy,
                    d_FloodCells_Default->qy,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qy for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_timeStep(unsigned int index)
 * Gets the value of the timeStep variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timeStep
 */
__host__ double get_FloodCell_Default_variable_timeStep(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_timeStep_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->timeStep,
                    d_FloodCells_Default->timeStep,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_timeStep_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->timeStep[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access timeStep for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_minh_loc(unsigned int index)
 * Gets the value of the minh_loc variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable minh_loc
 */
__host__ double get_FloodCell_Default_variable_minh_loc(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_minh_loc_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->minh_loc,
                    d_FloodCells_Default->minh_loc,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_minh_loc_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->minh_loc[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access minh_loc for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_hFace_E(unsigned int index)
 * Gets the value of the hFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_E
 */
__host__ double get_FloodCell_Default_variable_hFace_E(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_hFace_E_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->hFace_E,
                    d_FloodCells_Default->hFace_E,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_hFace_E_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->hFace_E[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hFace_E for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_etFace_E(unsigned int index)
 * Gets the value of the etFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_E
 */
__host__ double get_FloodCell_Default_variable_etFace_E(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_etFace_E_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->etFace_E,
                    d_FloodCells_Default->etFace_E,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_etFace_E_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->etFace_E[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access etFace_E for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qxFace_E(unsigned int index)
 * Gets the value of the qxFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_E
 */
__host__ double get_FloodCell_Default_variable_qxFace_E(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qxFace_E_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qxFace_E,
                    d_FloodCells_Default->qxFace_E,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qxFace_E_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qxFace_E[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qxFace_E for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qyFace_E(unsigned int index)
 * Gets the value of the qyFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_E
 */
__host__ double get_FloodCell_Default_variable_qyFace_E(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qyFace_E_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qyFace_E,
                    d_FloodCells_Default->qyFace_E,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qyFace_E_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qyFace_E[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qyFace_E for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_hFace_W(unsigned int index)
 * Gets the value of the hFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_W
 */
__host__ double get_FloodCell_Default_variable_hFace_W(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_hFace_W_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->hFace_W,
                    d_FloodCells_Default->hFace_W,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_hFace_W_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->hFace_W[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hFace_W for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_etFace_W(unsigned int index)
 * Gets the value of the etFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_W
 */
__host__ double get_FloodCell_Default_variable_etFace_W(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_etFace_W_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->etFace_W,
                    d_FloodCells_Default->etFace_W,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_etFace_W_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->etFace_W[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access etFace_W for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qxFace_W(unsigned int index)
 * Gets the value of the qxFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_W
 */
__host__ double get_FloodCell_Default_variable_qxFace_W(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qxFace_W_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qxFace_W,
                    d_FloodCells_Default->qxFace_W,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qxFace_W_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qxFace_W[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qxFace_W for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qyFace_W(unsigned int index)
 * Gets the value of the qyFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_W
 */
__host__ double get_FloodCell_Default_variable_qyFace_W(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qyFace_W_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qyFace_W,
                    d_FloodCells_Default->qyFace_W,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qyFace_W_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qyFace_W[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qyFace_W for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_hFace_N(unsigned int index)
 * Gets the value of the hFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_N
 */
__host__ double get_FloodCell_Default_variable_hFace_N(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_hFace_N_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->hFace_N,
                    d_FloodCells_Default->hFace_N,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_hFace_N_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->hFace_N[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hFace_N for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_etFace_N(unsigned int index)
 * Gets the value of the etFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_N
 */
__host__ double get_FloodCell_Default_variable_etFace_N(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_etFace_N_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->etFace_N,
                    d_FloodCells_Default->etFace_N,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_etFace_N_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->etFace_N[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access etFace_N for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qxFace_N(unsigned int index)
 * Gets the value of the qxFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_N
 */
__host__ double get_FloodCell_Default_variable_qxFace_N(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qxFace_N_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qxFace_N,
                    d_FloodCells_Default->qxFace_N,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qxFace_N_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qxFace_N[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qxFace_N for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qyFace_N(unsigned int index)
 * Gets the value of the qyFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_N
 */
__host__ double get_FloodCell_Default_variable_qyFace_N(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qyFace_N_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qyFace_N,
                    d_FloodCells_Default->qyFace_N,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qyFace_N_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qyFace_N[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qyFace_N for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_hFace_S(unsigned int index)
 * Gets the value of the hFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_S
 */
__host__ double get_FloodCell_Default_variable_hFace_S(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_hFace_S_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->hFace_S,
                    d_FloodCells_Default->hFace_S,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_hFace_S_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->hFace_S[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hFace_S for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_etFace_S(unsigned int index)
 * Gets the value of the etFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_S
 */
__host__ double get_FloodCell_Default_variable_etFace_S(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_etFace_S_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->etFace_S,
                    d_FloodCells_Default->etFace_S,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_etFace_S_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->etFace_S[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access etFace_S for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qxFace_S(unsigned int index)
 * Gets the value of the qxFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_S
 */
__host__ double get_FloodCell_Default_variable_qxFace_S(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qxFace_S_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qxFace_S,
                    d_FloodCells_Default->qxFace_S,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qxFace_S_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qxFace_S[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qxFace_S for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** double get_FloodCell_Default_variable_qyFace_S(unsigned int index)
 * Gets the value of the qyFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_S
 */
__host__ double get_FloodCell_Default_variable_qyFace_S(unsigned int index){
    unsigned int count = get_agent_FloodCell_Default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_FloodCells_Default_variable_qyFace_S_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_FloodCells_Default->qyFace_S,
                    d_FloodCells_Default->qyFace_S,
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_FloodCells_Default_variable_qyFace_S_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_FloodCells_Default->qyFace_S[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access qyFace_S for the %u th member of FloodCell_Default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/*  Analytics Functions */

int reduce_FloodCell_Default_inDomain_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->inDomain),  thrust::device_pointer_cast(d_FloodCells_Default->inDomain) + h_xmachine_memory_FloodCell_Default_count);
}

int count_FloodCell_Default_inDomain_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_FloodCells_Default->inDomain),  thrust::device_pointer_cast(d_FloodCells_Default->inDomain) + h_xmachine_memory_FloodCell_Default_count, count_value);
}
int min_FloodCell_Default_inDomain_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->inDomain);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_FloodCell_Default_inDomain_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->inDomain);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_FloodCell_Default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->x),  thrust::device_pointer_cast(d_FloodCells_Default->x) + h_xmachine_memory_FloodCell_Default_count);
}

int count_FloodCell_Default_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_FloodCells_Default->x),  thrust::device_pointer_cast(d_FloodCells_Default->x) + h_xmachine_memory_FloodCell_Default_count, count_value);
}
int min_FloodCell_Default_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_FloodCell_Default_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_FloodCell_Default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->y),  thrust::device_pointer_cast(d_FloodCells_Default->y) + h_xmachine_memory_FloodCell_Default_count);
}

int count_FloodCell_Default_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_FloodCells_Default->y),  thrust::device_pointer_cast(d_FloodCells_Default->y) + h_xmachine_memory_FloodCell_Default_count, count_value);
}
int min_FloodCell_Default_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_FloodCell_Default_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_z0_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->z0),  thrust::device_pointer_cast(d_FloodCells_Default->z0) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_z0_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->z0);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_z0_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->z0);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_h_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->h),  thrust::device_pointer_cast(d_FloodCells_Default->h) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_h_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->h);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_h_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->h);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qx),  thrust::device_pointer_cast(d_FloodCells_Default->qx) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qx_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qx_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qy),  thrust::device_pointer_cast(d_FloodCells_Default->qy) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qy_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qy_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_timeStep_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->timeStep),  thrust::device_pointer_cast(d_FloodCells_Default->timeStep) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_timeStep_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->timeStep);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_timeStep_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->timeStep);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_minh_loc_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->minh_loc),  thrust::device_pointer_cast(d_FloodCells_Default->minh_loc) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_minh_loc_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->minh_loc);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_minh_loc_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->minh_loc);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_hFace_E_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->hFace_E),  thrust::device_pointer_cast(d_FloodCells_Default->hFace_E) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_hFace_E_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_E);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_hFace_E_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_E);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_etFace_E_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->etFace_E),  thrust::device_pointer_cast(d_FloodCells_Default->etFace_E) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_etFace_E_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_E);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_etFace_E_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_E);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qxFace_E_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qxFace_E),  thrust::device_pointer_cast(d_FloodCells_Default->qxFace_E) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qxFace_E_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_E);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qxFace_E_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_E);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qyFace_E_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qyFace_E),  thrust::device_pointer_cast(d_FloodCells_Default->qyFace_E) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qyFace_E_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_E);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qyFace_E_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_E);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_hFace_W_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->hFace_W),  thrust::device_pointer_cast(d_FloodCells_Default->hFace_W) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_hFace_W_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_W);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_hFace_W_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_W);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_etFace_W_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->etFace_W),  thrust::device_pointer_cast(d_FloodCells_Default->etFace_W) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_etFace_W_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_W);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_etFace_W_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_W);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qxFace_W_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qxFace_W),  thrust::device_pointer_cast(d_FloodCells_Default->qxFace_W) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qxFace_W_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_W);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qxFace_W_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_W);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qyFace_W_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qyFace_W),  thrust::device_pointer_cast(d_FloodCells_Default->qyFace_W) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qyFace_W_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_W);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qyFace_W_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_W);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_hFace_N_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->hFace_N),  thrust::device_pointer_cast(d_FloodCells_Default->hFace_N) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_hFace_N_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_N);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_hFace_N_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_N);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_etFace_N_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->etFace_N),  thrust::device_pointer_cast(d_FloodCells_Default->etFace_N) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_etFace_N_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_N);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_etFace_N_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_N);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qxFace_N_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qxFace_N),  thrust::device_pointer_cast(d_FloodCells_Default->qxFace_N) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qxFace_N_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_N);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qxFace_N_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_N);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qyFace_N_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qyFace_N),  thrust::device_pointer_cast(d_FloodCells_Default->qyFace_N) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qyFace_N_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_N);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qyFace_N_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_N);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_hFace_S_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->hFace_S),  thrust::device_pointer_cast(d_FloodCells_Default->hFace_S) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_hFace_S_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_S);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_hFace_S_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->hFace_S);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_etFace_S_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->etFace_S),  thrust::device_pointer_cast(d_FloodCells_Default->etFace_S) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_etFace_S_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_S);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_etFace_S_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->etFace_S);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qxFace_S_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qxFace_S),  thrust::device_pointer_cast(d_FloodCells_Default->qxFace_S) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qxFace_S_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_S);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qxFace_S_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qxFace_S);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double reduce_FloodCell_Default_qyFace_S_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_FloodCells_Default->qyFace_S),  thrust::device_pointer_cast(d_FloodCells_Default->qyFace_S) + h_xmachine_memory_FloodCell_Default_count);
}

double min_FloodCell_Default_qyFace_S_variable(){
    //min in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_S);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
double max_FloodCell_Default_qyFace_S_variable(){
    //max in default stream
    thrust::device_ptr<double> thrust_ptr = thrust::device_pointer_cast(d_FloodCells_Default->qyFace_S);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_FloodCell_Default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int FloodCell_PrepareWetDry_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** FloodCell_PrepareWetDry
 * Agent function prototype for PrepareWetDry function of FloodCell agent
 */
void FloodCell_PrepareWetDry(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_FloodCell_Default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_FloodCell_Default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_FloodCell_list* FloodCells_Default_temp = d_FloodCells;
	d_FloodCells = d_FloodCells_Default;
	d_FloodCells_Default = FloodCells_Default_temp;
	//set working count to current state count
	h_xmachine_memory_FloodCell_count = h_xmachine_memory_FloodCell_Default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_count, &h_xmachine_memory_FloodCell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_FloodCell_Default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_PrepareWetDry, FloodCell_PrepareWetDry_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = FloodCell_PrepareWetDry_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (PrepareWetDry)
	//Reallocate   : false
	//Input        : 
	//Output       : WetDryMessage
	//Agent Output : 
	GPUFLAME_PrepareWetDry<<<g, b, sm_size, stream>>>(d_FloodCells, d_WetDryMessages);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	FloodCells_Default_temp = d_FloodCells_Default;
	d_FloodCells_Default = d_FloodCells;
	d_FloodCells = FloodCells_Default_temp;
    //set current state count
	h_xmachine_memory_FloodCell_Default_count = h_xmachine_memory_FloodCell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int FloodCell_ProcessWetDryMessage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_WetDryMessage_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_WetDryMessage)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** FloodCell_ProcessWetDryMessage
 * Agent function prototype for ProcessWetDryMessage function of FloodCell agent
 */
void FloodCell_ProcessWetDryMessage(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_FloodCell_Default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_FloodCell_Default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_FloodCell_list* FloodCells_Default_temp = d_FloodCells;
	d_FloodCells = d_FloodCells_Default;
	d_FloodCells_Default = FloodCells_Default_temp;
	//set working count to current state count
	h_xmachine_memory_FloodCell_count = h_xmachine_memory_FloodCell_Default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_count, &h_xmachine_memory_FloodCell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_FloodCell_Default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ProcessWetDryMessage, FloodCell_ProcessWetDryMessage_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = FloodCell_ProcessWetDryMessage_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_WetDryMessage_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_WetDryMessage_inDomain_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_WetDryMessage_inDomain_byte_offset, tex_xmachine_message_WetDryMessage_inDomain, d_WetDryMessages->inDomain, sizeof(int)*xmachine_message_WetDryMessage_MAX));
	h_tex_xmachine_message_WetDryMessage_inDomain_offset = (int)tex_xmachine_message_WetDryMessage_inDomain_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_WetDryMessage_inDomain_offset, &h_tex_xmachine_message_WetDryMessage_inDomain_offset, sizeof(int)));
	size_t tex_xmachine_message_WetDryMessage_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_WetDryMessage_x_byte_offset, tex_xmachine_message_WetDryMessage_x, d_WetDryMessages->x, sizeof(int)*xmachine_message_WetDryMessage_MAX));
	h_tex_xmachine_message_WetDryMessage_x_offset = (int)tex_xmachine_message_WetDryMessage_x_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_WetDryMessage_x_offset, &h_tex_xmachine_message_WetDryMessage_x_offset, sizeof(int)));
	size_t tex_xmachine_message_WetDryMessage_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_WetDryMessage_y_byte_offset, tex_xmachine_message_WetDryMessage_y, d_WetDryMessages->y, sizeof(int)*xmachine_message_WetDryMessage_MAX));
	h_tex_xmachine_message_WetDryMessage_y_offset = (int)tex_xmachine_message_WetDryMessage_y_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_WetDryMessage_y_offset, &h_tex_xmachine_message_WetDryMessage_y_offset, sizeof(int)));
	size_t tex_xmachine_message_WetDryMessage_min_hloc_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_WetDryMessage_min_hloc_byte_offset, tex_xmachine_message_WetDryMessage_min_hloc, d_WetDryMessages->min_hloc, sizeof(double)*xmachine_message_WetDryMessage_MAX));
	h_tex_xmachine_message_WetDryMessage_min_hloc_offset = (int)tex_xmachine_message_WetDryMessage_min_hloc_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_WetDryMessage_min_hloc_offset, &h_tex_xmachine_message_WetDryMessage_min_hloc_offset, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (ProcessWetDryMessage)
	//Reallocate   : false
	//Input        : WetDryMessage
	//Output       : 
	//Agent Output : 
	GPUFLAME_ProcessWetDryMessage<<<g, b, sm_size, stream>>>(d_FloodCells, d_WetDryMessages);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_WetDryMessage_inDomain));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_WetDryMessage_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_WetDryMessage_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_WetDryMessage_min_hloc));
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	FloodCells_Default_temp = d_FloodCells_Default;
	d_FloodCells_Default = d_FloodCells;
	d_FloodCells = FloodCells_Default_temp;
    //set current state count
	h_xmachine_memory_FloodCell_Default_count = h_xmachine_memory_FloodCell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int FloodCell_PrepareSpaceOperator_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** FloodCell_PrepareSpaceOperator
 * Agent function prototype for PrepareSpaceOperator function of FloodCell agent
 */
void FloodCell_PrepareSpaceOperator(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_FloodCell_Default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_FloodCell_Default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_FloodCell_list* FloodCells_Default_temp = d_FloodCells;
	d_FloodCells = d_FloodCells_Default;
	d_FloodCells_Default = FloodCells_Default_temp;
	//set working count to current state count
	h_xmachine_memory_FloodCell_count = h_xmachine_memory_FloodCell_Default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_count, &h_xmachine_memory_FloodCell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_FloodCell_Default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_PrepareSpaceOperator, FloodCell_PrepareSpaceOperator_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = FloodCell_PrepareSpaceOperator_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (PrepareSpaceOperator)
	//Reallocate   : false
	//Input        : 
	//Output       : SpaceOperatorMessage
	//Agent Output : 
	GPUFLAME_PrepareSpaceOperator<<<g, b, sm_size, stream>>>(d_FloodCells, d_SpaceOperatorMessages);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	FloodCells_Default_temp = d_FloodCells_Default;
	d_FloodCells_Default = d_FloodCells;
	d_FloodCells = FloodCells_Default_temp;
    //set current state count
	h_xmachine_memory_FloodCell_Default_count = h_xmachine_memory_FloodCell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int FloodCell_ProcessSpaceOperatorMessage_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_SpaceOperatorMessage_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_SpaceOperatorMessage)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** FloodCell_ProcessSpaceOperatorMessage
 * Agent function prototype for ProcessSpaceOperatorMessage function of FloodCell agent
 */
void FloodCell_ProcessSpaceOperatorMessage(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_FloodCell_Default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_FloodCell_Default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_FloodCell_list* FloodCells_Default_temp = d_FloodCells;
	d_FloodCells = d_FloodCells_Default;
	d_FloodCells_Default = FloodCells_Default_temp;
	//set working count to current state count
	h_xmachine_memory_FloodCell_count = h_xmachine_memory_FloodCell_Default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_count, &h_xmachine_memory_FloodCell_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_FloodCell_Default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_ProcessSpaceOperatorMessage, FloodCell_ProcessSpaceOperatorMessage_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = FloodCell_ProcessSpaceOperatorMessage_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_SpaceOperatorMessage_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_SpaceOperatorMessage_inDomain_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_inDomain_byte_offset, tex_xmachine_message_SpaceOperatorMessage_inDomain, d_SpaceOperatorMessages->inDomain, sizeof(int)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset = (int)tex_xmachine_message_SpaceOperatorMessage_inDomain_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset, &h_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_x_byte_offset, tex_xmachine_message_SpaceOperatorMessage_x, d_SpaceOperatorMessages->x, sizeof(int)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_x_offset = (int)tex_xmachine_message_SpaceOperatorMessage_x_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_x_offset, &h_tex_xmachine_message_SpaceOperatorMessage_x_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_y_byte_offset, tex_xmachine_message_SpaceOperatorMessage_y, d_SpaceOperatorMessages->y, sizeof(int)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_y_offset = (int)tex_xmachine_message_SpaceOperatorMessage_y_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_y_offset, &h_tex_xmachine_message_SpaceOperatorMessage_y_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_hFace_E_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_hFace_E_byte_offset, tex_xmachine_message_SpaceOperatorMessage_hFace_E, d_SpaceOperatorMessages->hFace_E, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset = (int)tex_xmachine_message_SpaceOperatorMessage_hFace_E_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset, &h_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_etFace_E_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_etFace_E_byte_offset, tex_xmachine_message_SpaceOperatorMessage_etFace_E, d_SpaceOperatorMessages->etFace_E, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset = (int)tex_xmachine_message_SpaceOperatorMessage_etFace_E_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset, &h_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_X_E, d_SpaceOperatorMessages->qFace_X_E, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E, d_SpaceOperatorMessages->qFace_Y_E, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_hFace_W_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_hFace_W_byte_offset, tex_xmachine_message_SpaceOperatorMessage_hFace_W, d_SpaceOperatorMessages->hFace_W, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset = (int)tex_xmachine_message_SpaceOperatorMessage_hFace_W_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset, &h_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_etFace_W_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_etFace_W_byte_offset, tex_xmachine_message_SpaceOperatorMessage_etFace_W, d_SpaceOperatorMessages->etFace_W, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset = (int)tex_xmachine_message_SpaceOperatorMessage_etFace_W_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset, &h_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_X_W, d_SpaceOperatorMessages->qFace_X_W, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W, d_SpaceOperatorMessages->qFace_Y_W, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_hFace_N_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_hFace_N_byte_offset, tex_xmachine_message_SpaceOperatorMessage_hFace_N, d_SpaceOperatorMessages->hFace_N, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset = (int)tex_xmachine_message_SpaceOperatorMessage_hFace_N_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset, &h_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_etFace_N_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_etFace_N_byte_offset, tex_xmachine_message_SpaceOperatorMessage_etFace_N, d_SpaceOperatorMessages->etFace_N, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset = (int)tex_xmachine_message_SpaceOperatorMessage_etFace_N_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset, &h_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_X_N, d_SpaceOperatorMessages->qFace_X_N, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N, d_SpaceOperatorMessages->qFace_Y_N, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_hFace_S_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_hFace_S_byte_offset, tex_xmachine_message_SpaceOperatorMessage_hFace_S, d_SpaceOperatorMessages->hFace_S, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset = (int)tex_xmachine_message_SpaceOperatorMessage_hFace_S_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset, &h_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_etFace_S_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_etFace_S_byte_offset, tex_xmachine_message_SpaceOperatorMessage_etFace_S, d_SpaceOperatorMessages->etFace_S, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset = (int)tex_xmachine_message_SpaceOperatorMessage_etFace_S_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset, &h_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_X_S, d_SpaceOperatorMessages->qFace_X_S, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset, sizeof(int)));
	size_t tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_byte_offset, tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S, d_SpaceOperatorMessages->qFace_Y_S, sizeof(double)*xmachine_message_SpaceOperatorMessage_MAX));
	h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset = (int)tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_byte_offset / sizeof(double);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset, &h_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (ProcessSpaceOperatorMessage)
	//Reallocate   : false
	//Input        : SpaceOperatorMessage
	//Output       : 
	//Agent Output : 
	GPUFLAME_ProcessSpaceOperatorMessage<<<g, b, sm_size, stream>>>(d_FloodCells, d_SpaceOperatorMessages);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_inDomain));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_hFace_E));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_etFace_E));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_X_E));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_hFace_W));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_etFace_W));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_X_W));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_hFace_N));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_etFace_N));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_X_N));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_hFace_S));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_etFace_S));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_X_S));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S));
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	FloodCells_Default_temp = d_FloodCells_Default;
	d_FloodCells_Default = d_FloodCells;
	d_FloodCells = FloodCells_Default_temp;
    //set current state count
	h_xmachine_memory_FloodCell_Default_count = h_xmachine_memory_FloodCell_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_FloodCell_Default_count, &h_xmachine_memory_FloodCell_Default_count, sizeof(int)));	
	
	
}


 
extern void reset_FloodCell_Default_count()
{
    h_xmachine_memory_FloodCell_Default_count = 0;
}
