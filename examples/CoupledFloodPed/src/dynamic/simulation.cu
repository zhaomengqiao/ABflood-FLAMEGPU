
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

/* agent Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_agent_list* d_agents;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_agent_list* d_agents_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_agent_list* d_agents_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_agent_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_agent_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_agent_values;  /**< Agent sort identifiers value */

/* agent state variables */
xmachine_memory_agent_list* h_agents_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_agent_list* d_agents_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_agent_default_count;   /**< Agent population size counter */ 

/* navmap Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_navmap_list* d_navmaps;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_navmap_list* d_navmaps_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_navmap_list* d_navmaps_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_navmap_count;   /**< Agent population size counter */ 
int h_xmachine_memory_navmap_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_navmap_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_navmap_values;  /**< Agent sort identifiers value */

/* navmap state variables */
xmachine_memory_navmap_list* h_navmaps_static;      /**< Pointer to agent list (population) on host*/
xmachine_memory_navmap_list* d_navmaps_static;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_navmap_static_count;   /**< Agent population size counter */ 


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
unsigned int h_agents_default_variable_x_data_iteration;
unsigned int h_agents_default_variable_y_data_iteration;
unsigned int h_agents_default_variable_velx_data_iteration;
unsigned int h_agents_default_variable_vely_data_iteration;
unsigned int h_agents_default_variable_steer_x_data_iteration;
unsigned int h_agents_default_variable_steer_y_data_iteration;
unsigned int h_agents_default_variable_height_data_iteration;
unsigned int h_agents_default_variable_exit_no_data_iteration;
unsigned int h_agents_default_variable_speed_data_iteration;
unsigned int h_agents_default_variable_lod_data_iteration;
unsigned int h_agents_default_variable_animate_data_iteration;
unsigned int h_agents_default_variable_animate_dir_data_iteration;
unsigned int h_navmaps_static_variable_x_data_iteration;
unsigned int h_navmaps_static_variable_y_data_iteration;
unsigned int h_navmaps_static_variable_exit_no_data_iteration;
unsigned int h_navmaps_static_variable_height_data_iteration;
unsigned int h_navmaps_static_variable_collision_x_data_iteration;
unsigned int h_navmaps_static_variable_collision_y_data_iteration;
unsigned int h_navmaps_static_variable_exit0_x_data_iteration;
unsigned int h_navmaps_static_variable_exit0_y_data_iteration;
unsigned int h_navmaps_static_variable_exit1_x_data_iteration;
unsigned int h_navmaps_static_variable_exit1_y_data_iteration;
unsigned int h_navmaps_static_variable_exit2_x_data_iteration;
unsigned int h_navmaps_static_variable_exit2_y_data_iteration;
unsigned int h_navmaps_static_variable_exit3_x_data_iteration;
unsigned int h_navmaps_static_variable_exit3_y_data_iteration;
unsigned int h_navmaps_static_variable_exit4_x_data_iteration;
unsigned int h_navmaps_static_variable_exit4_y_data_iteration;
unsigned int h_navmaps_static_variable_exit5_x_data_iteration;
unsigned int h_navmaps_static_variable_exit5_y_data_iteration;
unsigned int h_navmaps_static_variable_exit6_x_data_iteration;
unsigned int h_navmaps_static_variable_exit6_y_data_iteration;


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
/* pedestrian_location Message variables */
xmachine_message_pedestrian_location_list* h_pedestrian_locations;         /**< Pointer to message list on host*/
xmachine_message_pedestrian_location_list* d_pedestrian_locations;         /**< Pointer to message list on device*/
xmachine_message_pedestrian_location_list* d_pedestrian_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_pedestrian_location_count;         /**< message list counter*/
int h_message_pedestrian_location_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_pedestrian_location_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_pedestrian_location_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_scan_tmp_memory_pedestrian_location;
    size_t scan_tmp_bytes_pedestrian_location;
#else
	uint * d_xmachine_message_pedestrian_location_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_pedestrian_location_values;  /**< message sort identifier values */
#endif
xmachine_message_pedestrian_location_PBM * d_pedestrian_location_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_pedestrian_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_pedestrian_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_pedestrian_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_pedestrian_location_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_pedestrian_location_x_offset;
int h_tex_xmachine_message_pedestrian_location_y_offset;
int h_tex_xmachine_message_pedestrian_location_z_offset;
int h_tex_xmachine_message_pedestrian_location_pbm_start_offset;
int h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset;

/* navmap_cell Message variables */
xmachine_message_navmap_cell_list* h_navmap_cells;         /**< Pointer to message list on host*/
xmachine_message_navmap_cell_list* d_navmap_cells;         /**< Pointer to message list on device*/
xmachine_message_navmap_cell_list* d_navmap_cells_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_navmap_cell_range;     /**< range of the discrete message*/
int h_message_navmap_cell_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_navmap_cell_x_offset;
int h_tex_xmachine_message_navmap_cell_y_offset;
int h_tex_xmachine_message_navmap_cell_exit_no_offset;
int h_tex_xmachine_message_navmap_cell_height_offset;
int h_tex_xmachine_message_navmap_cell_collision_x_offset;
int h_tex_xmachine_message_navmap_cell_collision_y_offset;
int h_tex_xmachine_message_navmap_cell_exit0_x_offset;
int h_tex_xmachine_message_navmap_cell_exit0_y_offset;
int h_tex_xmachine_message_navmap_cell_exit1_x_offset;
int h_tex_xmachine_message_navmap_cell_exit1_y_offset;
int h_tex_xmachine_message_navmap_cell_exit2_x_offset;
int h_tex_xmachine_message_navmap_cell_exit2_y_offset;
int h_tex_xmachine_message_navmap_cell_exit3_x_offset;
int h_tex_xmachine_message_navmap_cell_exit3_y_offset;
int h_tex_xmachine_message_navmap_cell_exit4_x_offset;
int h_tex_xmachine_message_navmap_cell_exit4_y_offset;
int h_tex_xmachine_message_navmap_cell_exit5_x_offset;
int h_tex_xmachine_message_navmap_cell_exit5_y_offset;
int h_tex_xmachine_message_navmap_cell_exit6_x_offset;
int h_tex_xmachine_message_navmap_cell_exit6_y_offset;
  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_FloodCell;
size_t temp_scan_storage_bytes_FloodCell;

void * d_temp_scan_storage_agent;
size_t temp_scan_storage_bytes_agent;

void * d_temp_scan_storage_navmap;
size_t temp_scan_storage_bytes_navmap;


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

/** agent_output_pedestrian_location
 * Agent function prototype for output_pedestrian_location function of agent agent
 */
void agent_output_pedestrian_location(cudaStream_t &stream);

/** agent_avoid_pedestrians
 * Agent function prototype for avoid_pedestrians function of agent agent
 */
void agent_avoid_pedestrians(cudaStream_t &stream);

/** agent_force_flow
 * Agent function prototype for force_flow function of agent agent
 */
void agent_force_flow(cudaStream_t &stream);

/** agent_move
 * Agent function prototype for move function of agent agent
 */
void agent_move(cudaStream_t &stream);

/** navmap_output_navmap_cells
 * Agent function prototype for output_navmap_cells function of navmap agent
 */
void navmap_output_navmap_cells(cudaStream_t &stream);

/** navmap_generate_pedestrians
 * Agent function prototype for generate_pedestrians function of navmap agent
 */
void navmap_generate_pedestrians(cudaStream_t &stream);

  
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
    h_agents_default_variable_x_data_iteration = 0;
    h_agents_default_variable_y_data_iteration = 0;
    h_agents_default_variable_velx_data_iteration = 0;
    h_agents_default_variable_vely_data_iteration = 0;
    h_agents_default_variable_steer_x_data_iteration = 0;
    h_agents_default_variable_steer_y_data_iteration = 0;
    h_agents_default_variable_height_data_iteration = 0;
    h_agents_default_variable_exit_no_data_iteration = 0;
    h_agents_default_variable_speed_data_iteration = 0;
    h_agents_default_variable_lod_data_iteration = 0;
    h_agents_default_variable_animate_data_iteration = 0;
    h_agents_default_variable_animate_dir_data_iteration = 0;
    h_navmaps_static_variable_x_data_iteration = 0;
    h_navmaps_static_variable_y_data_iteration = 0;
    h_navmaps_static_variable_exit_no_data_iteration = 0;
    h_navmaps_static_variable_height_data_iteration = 0;
    h_navmaps_static_variable_collision_x_data_iteration = 0;
    h_navmaps_static_variable_collision_y_data_iteration = 0;
    h_navmaps_static_variable_exit0_x_data_iteration = 0;
    h_navmaps_static_variable_exit0_y_data_iteration = 0;
    h_navmaps_static_variable_exit1_x_data_iteration = 0;
    h_navmaps_static_variable_exit1_y_data_iteration = 0;
    h_navmaps_static_variable_exit2_x_data_iteration = 0;
    h_navmaps_static_variable_exit2_y_data_iteration = 0;
    h_navmaps_static_variable_exit3_x_data_iteration = 0;
    h_navmaps_static_variable_exit3_y_data_iteration = 0;
    h_navmaps_static_variable_exit4_x_data_iteration = 0;
    h_navmaps_static_variable_exit4_y_data_iteration = 0;
    h_navmaps_static_variable_exit5_x_data_iteration = 0;
    h_navmaps_static_variable_exit5_y_data_iteration = 0;
    h_navmaps_static_variable_exit6_x_data_iteration = 0;
    h_navmaps_static_variable_exit6_y_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_FloodCell_SoA_size = sizeof(xmachine_memory_FloodCell_list);
	h_FloodCells_Default = (xmachine_memory_FloodCell_list*)malloc(xmachine_FloodCell_SoA_size);
	int xmachine_agent_SoA_size = sizeof(xmachine_memory_agent_list);
	h_agents_default = (xmachine_memory_agent_list*)malloc(xmachine_agent_SoA_size);
	int xmachine_navmap_SoA_size = sizeof(xmachine_memory_navmap_list);
	h_navmaps_static = (xmachine_memory_navmap_list*)malloc(xmachine_navmap_SoA_size);

	/* Message memory allocation (CPU) */
	int message_WetDryMessage_SoA_size = sizeof(xmachine_message_WetDryMessage_list);
	h_WetDryMessages = (xmachine_message_WetDryMessage_list*)malloc(message_WetDryMessage_SoA_size);
	int message_SpaceOperatorMessage_SoA_size = sizeof(xmachine_message_SpaceOperatorMessage_list);
	h_SpaceOperatorMessages = (xmachine_message_SpaceOperatorMessage_list*)malloc(message_SpaceOperatorMessage_SoA_size);
	int message_pedestrian_location_SoA_size = sizeof(xmachine_message_pedestrian_location_list);
	h_pedestrian_locations = (xmachine_message_pedestrian_location_list*)malloc(message_pedestrian_location_SoA_size);
	int message_navmap_cell_SoA_size = sizeof(xmachine_message_navmap_cell_list);
	h_navmap_cells = (xmachine_message_navmap_cell_list*)malloc(message_navmap_cell_SoA_size);

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
	
			
	/* Set spatial partitioning pedestrian_location message variables (min_bounds, max_bounds)*/
	h_message_pedestrian_location_radius = (float)0.025;
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_radius, &h_message_pedestrian_location_radius, sizeof(float)));	
	    h_message_pedestrian_location_min_bounds = glm::vec3((float)-1.0, (float)-1.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_min_bounds, &h_message_pedestrian_location_min_bounds, sizeof(glm::vec3)));	
	h_message_pedestrian_location_max_bounds = glm::vec3((float)1.0, (float)1.0, (float)0.025);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_max_bounds, &h_message_pedestrian_location_max_bounds, sizeof(glm::vec3)));	
	h_message_pedestrian_location_partitionDim.x = (int)ceil((h_message_pedestrian_location_max_bounds.x - h_message_pedestrian_location_min_bounds.x)/h_message_pedestrian_location_radius);
	h_message_pedestrian_location_partitionDim.y = (int)ceil((h_message_pedestrian_location_max_bounds.y - h_message_pedestrian_location_min_bounds.y)/h_message_pedestrian_location_radius);
	h_message_pedestrian_location_partitionDim.z = (int)ceil((h_message_pedestrian_location_max_bounds.z - h_message_pedestrian_location_min_bounds.z)/h_message_pedestrian_location_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_partitionDim, &h_message_pedestrian_location_partitionDim, sizeof(glm::ivec3)));	
	
	
	/* Set discrete navmap_cell message variables (range, width)*/
	h_message_navmap_cell_range = 0; //from xml
	h_message_navmap_cell_width = (int)floor(sqrt((float)xmachine_message_navmap_cell_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_navmap_cell_MAX)){
		printf("ERROR: navmap_cell message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_navmap_cell_range, &h_message_navmap_cell_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_navmap_cell_width, &h_message_navmap_cell_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_FloodCell_MAX)){
		printf("ERROR: FloodCells agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_FloodCell_pop_width = (int)sqrt(xmachine_memory_FloodCell_MAX);
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_navmap_MAX)){
		printf("ERROR: navmaps agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_navmap_pop_width = (int)sqrt(xmachine_memory_navmap_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_FloodCells_Default, &h_xmachine_memory_FloodCell_Default_count, h_agents_default, &h_xmachine_memory_agent_default_count, h_navmaps_static, &h_xmachine_memory_navmap_static_count);
	

    PROFILE_PUSH_RANGE("allocate device");
	
	/* FloodCell Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells, xmachine_FloodCell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells_swap, xmachine_FloodCell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells_new, xmachine_FloodCell_SoA_size));
    
	/* Default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_FloodCells_Default, xmachine_FloodCell_SoA_size));
	gpuErrchk( cudaMemcpy( d_FloodCells_Default, h_FloodCells_Default, xmachine_FloodCell_SoA_size, cudaMemcpyHostToDevice));
    
	/* agent Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_swap, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_new, xmachine_agent_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_agent_keys, xmachine_memory_agent_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_agent_values, xmachine_memory_agent_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents_default, xmachine_agent_SoA_size));
	gpuErrchk( cudaMemcpy( d_agents_default, h_agents_default, xmachine_agent_SoA_size, cudaMemcpyHostToDevice));
    
	/* navmap Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmaps, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_swap, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_new, xmachine_navmap_SoA_size));
    
	/* static memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_static, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMemcpy( d_navmaps_static, h_navmaps_static, xmachine_navmap_SoA_size, cudaMemcpyHostToDevice));
    
	/* WetDryMessage Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_WetDryMessages, message_WetDryMessage_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_WetDryMessages_swap, message_WetDryMessage_SoA_size));
	gpuErrchk( cudaMemcpy( d_WetDryMessages, h_WetDryMessages, message_WetDryMessage_SoA_size, cudaMemcpyHostToDevice));
	
	/* SpaceOperatorMessage Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_SpaceOperatorMessages, message_SpaceOperatorMessage_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_SpaceOperatorMessages_swap, message_SpaceOperatorMessage_SoA_size));
	gpuErrchk( cudaMemcpy( d_SpaceOperatorMessages, h_SpaceOperatorMessages, message_SpaceOperatorMessage_SoA_size, cudaMemcpyHostToDevice));
	
	/* pedestrian_location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_locations, message_pedestrian_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_locations_swap, message_pedestrian_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_pedestrian_locations, h_pedestrian_locations, message_pedestrian_location_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_location_partition_matrix, sizeof(xmachine_message_pedestrian_location_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_local_bin_index, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_unsorted_index, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_scan_tmp_memory_pedestrian_location = nullptr;
    scan_tmp_bytes_pedestrian_location = 0;
    cub::DeviceScan::ExclusiveSum(
        d_scan_tmp_memory_pedestrian_location, 
        scan_tmp_bytes_pedestrian_location, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_pedestrian_location_grid_size
    );
    gpuErrchk(cudaMalloc(&d_scan_tmp_memory_pedestrian_location, scan_tmp_bytes_pedestrian_location));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_keys, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_values, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
#endif
	
	/* navmap_cell Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmap_cells, message_navmap_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmap_cells_swap, message_navmap_cell_SoA_size));
	gpuErrchk( cudaMemcpy( d_navmap_cells, h_navmap_cells, message_navmap_cell_SoA_size, cudaMemcpyHostToDevice));
		
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
    
    d_temp_scan_storage_agent = nullptr;
    temp_scan_storage_bytes_agent = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_agent_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_agent, temp_scan_storage_bytes_agent));
    
    d_temp_scan_storage_navmap = nullptr;
    temp_scan_storage_bytes_navmap = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_navmap, 
        temp_scan_storage_bytes_navmap, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_navmap_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_navmap, temp_scan_storage_bytes_navmap));
    

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
  gpuErrchk(cudaStreamCreate(&stream2));
  gpuErrchk(cudaStreamCreate(&stream3));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_FloodCell_Default_count: %u\n",get_agent_FloodCell_Default_count());
	
		printf("Init agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
		printf("Init agent_navmap_static_count: %u\n",get_agent_navmap_static_count());
	
#endif
} 


void sort_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_agent_default_count); 
	gridSize = (h_xmachine_memory_agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_agent_keys, d_xmachine_memory_agent_values, d_agents_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_agent_keys),  thrust::device_pointer_cast(d_xmachine_memory_agent_keys) + h_xmachine_memory_agent_default_count,  thrust::device_pointer_cast(d_xmachine_memory_agent_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agent_agents, no_sm, h_xmachine_memory_agent_default_count); 
	gridSize = (h_xmachine_memory_agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_agent_agents<<<gridSize, blockSize>>>(d_xmachine_memory_agent_values, d_agents_default, d_agents_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_agent_list* d_agents_temp = d_agents_default;
	d_agents_default = d_agents_swap;
	d_agents_swap = d_agents_temp;	
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
	
	/* agent Agent variables */
	gpuErrchk(cudaFree(d_agents));
	gpuErrchk(cudaFree(d_agents_swap));
	gpuErrchk(cudaFree(d_agents_new));
	
	free( h_agents_default);
	gpuErrchk(cudaFree(d_agents_default));
	
	/* navmap Agent variables */
	gpuErrchk(cudaFree(d_navmaps));
	gpuErrchk(cudaFree(d_navmaps_swap));
	gpuErrchk(cudaFree(d_navmaps_new));
	
	free( h_navmaps_static);
	gpuErrchk(cudaFree(d_navmaps_static));
	

	/* Message data free */
	
	/* WetDryMessage Message variables */
	free( h_WetDryMessages);
	gpuErrchk(cudaFree(d_WetDryMessages));
	gpuErrchk(cudaFree(d_WetDryMessages_swap));
	
	/* SpaceOperatorMessage Message variables */
	free( h_SpaceOperatorMessages);
	gpuErrchk(cudaFree(d_SpaceOperatorMessages));
	gpuErrchk(cudaFree(d_SpaceOperatorMessages_swap));
	
	/* pedestrian_location Message variables */
	free( h_pedestrian_locations);
	gpuErrchk(cudaFree(d_pedestrian_locations));
	gpuErrchk(cudaFree(d_pedestrian_locations_swap));
	gpuErrchk(cudaFree(d_pedestrian_location_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_unsorted_index));
    gpuErrchk(cudaFree(d_scan_tmp_memory_pedestrian_location));
    d_scan_tmp_memory_pedestrian_location = nullptr;
    scan_tmp_bytes_pedestrian_location = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_keys));
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_values));
#endif
	
	/* navmap_cell Message variables */
	free( h_navmap_cells);
	gpuErrchk(cudaFree(d_navmap_cells));
	gpuErrchk(cudaFree(d_navmap_cells_swap));
	

    /* Free temporary CUB memory */
    
    gpuErrchk(cudaFree(d_temp_scan_storage_FloodCell));
    d_temp_scan_storage_FloodCell = nullptr;
    temp_scan_storage_bytes_FloodCell = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_agent));
    d_temp_scan_storage_agent = nullptr;
    temp_scan_storage_bytes_agent = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_navmap));
    d_temp_scan_storage_navmap = nullptr;
    temp_scan_storage_bytes_navmap = 0;
    
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));
  gpuErrchk(cudaStreamDestroy(stream3));

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
	h_message_pedestrian_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_count, &h_message_pedestrian_location_count, sizeof(int)));
	

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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("navmap_generate_pedestrians");
	navmap_generate_pedestrians(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: navmap_generate_pedestrians = %f (ms)\n", instrument_milliseconds);
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_pedestrian_location");
	agent_output_pedestrian_location(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_pedestrian_location = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("navmap_output_navmap_cells");
	navmap_output_navmap_cells(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: navmap_output_navmap_cells = %f (ms)\n", instrument_milliseconds);
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_avoid_pedestrians");
	agent_avoid_pedestrians(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_avoid_pedestrians = %f (ms)\n", instrument_milliseconds);
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_force_flow");
	agent_force_flow(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_force_flow = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_move");
	agent_move(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_move = %f (ms)\n", instrument_milliseconds);
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
	
		printf("agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
		printf("agent_navmap_static_count: %u\n",get_agent_navmap_static_count());
	
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
double h_env_xmin;
double h_env_xmax;
double h_env_ymin;
double h_env_ymax;
double h_env_dt;
double h_env_DXL;
double h_env_DYL;
float h_env_EMMISION_RATE_EXIT1;
float h_env_EMMISION_RATE_EXIT2;
float h_env_EMMISION_RATE_EXIT3;
float h_env_EMMISION_RATE_EXIT4;
float h_env_EMMISION_RATE_EXIT5;
float h_env_EMMISION_RATE_EXIT6;
float h_env_EMMISION_RATE_EXIT7;
int h_env_EXIT1_PROBABILITY;
int h_env_EXIT2_PROBABILITY;
int h_env_EXIT3_PROBABILITY;
int h_env_EXIT4_PROBABILITY;
int h_env_EXIT5_PROBABILITY;
int h_env_EXIT6_PROBABILITY;
int h_env_EXIT7_PROBABILITY;
int h_env_EXIT1_STATE;
int h_env_EXIT2_STATE;
int h_env_EXIT3_STATE;
int h_env_EXIT4_STATE;
int h_env_EXIT5_STATE;
int h_env_EXIT6_STATE;
int h_env_EXIT7_STATE;
int h_env_EXIT1_CELL_COUNT;
int h_env_EXIT2_CELL_COUNT;
int h_env_EXIT3_CELL_COUNT;
int h_env_EXIT4_CELL_COUNT;
int h_env_EXIT5_CELL_COUNT;
int h_env_EXIT6_CELL_COUNT;
int h_env_EXIT7_CELL_COUNT;
float h_env_TIME_SCALER;
float h_env_STEER_WEIGHT;
float h_env_AVOID_WEIGHT;
float h_env_COLLISION_WEIGHT;
float h_env_GOAL_WEIGHT;


//constant setter
void set_xmin(double* h_xmin){
    gpuErrchk(cudaMemcpyToSymbol(xmin, h_xmin, sizeof(double)));
    memcpy(&h_env_xmin, h_xmin,sizeof(double));
}

//constant getter
const double* get_xmin(){
    return &h_env_xmin;
}



//constant setter
void set_xmax(double* h_xmax){
    gpuErrchk(cudaMemcpyToSymbol(xmax, h_xmax, sizeof(double)));
    memcpy(&h_env_xmax, h_xmax,sizeof(double));
}

//constant getter
const double* get_xmax(){
    return &h_env_xmax;
}



//constant setter
void set_ymin(double* h_ymin){
    gpuErrchk(cudaMemcpyToSymbol(ymin, h_ymin, sizeof(double)));
    memcpy(&h_env_ymin, h_ymin,sizeof(double));
}

//constant getter
const double* get_ymin(){
    return &h_env_ymin;
}



//constant setter
void set_ymax(double* h_ymax){
    gpuErrchk(cudaMemcpyToSymbol(ymax, h_ymax, sizeof(double)));
    memcpy(&h_env_ymax, h_ymax,sizeof(double));
}

//constant getter
const double* get_ymax(){
    return &h_env_ymax;
}



//constant setter
void set_dt(double* h_dt){
    gpuErrchk(cudaMemcpyToSymbol(dt, h_dt, sizeof(double)));
    memcpy(&h_env_dt, h_dt,sizeof(double));
}

//constant getter
const double* get_dt(){
    return &h_env_dt;
}



//constant setter
void set_DXL(double* h_DXL){
    gpuErrchk(cudaMemcpyToSymbol(DXL, h_DXL, sizeof(double)));
    memcpy(&h_env_DXL, h_DXL,sizeof(double));
}

//constant getter
const double* get_DXL(){
    return &h_env_DXL;
}



//constant setter
void set_DYL(double* h_DYL){
    gpuErrchk(cudaMemcpyToSymbol(DYL, h_DYL, sizeof(double)));
    memcpy(&h_env_DYL, h_DYL,sizeof(double));
}

//constant getter
const double* get_DYL(){
    return &h_env_DYL;
}



//constant setter
void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE_EXIT1){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT1, h_EMMISION_RATE_EXIT1, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT1, h_EMMISION_RATE_EXIT1,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT1(){
    return &h_env_EMMISION_RATE_EXIT1;
}



//constant setter
void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE_EXIT2){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT2, h_EMMISION_RATE_EXIT2, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT2, h_EMMISION_RATE_EXIT2,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT2(){
    return &h_env_EMMISION_RATE_EXIT2;
}



//constant setter
void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE_EXIT3){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT3, h_EMMISION_RATE_EXIT3, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT3, h_EMMISION_RATE_EXIT3,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT3(){
    return &h_env_EMMISION_RATE_EXIT3;
}



//constant setter
void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE_EXIT4){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT4, h_EMMISION_RATE_EXIT4, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT4, h_EMMISION_RATE_EXIT4,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT4(){
    return &h_env_EMMISION_RATE_EXIT4;
}



//constant setter
void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE_EXIT5){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT5, h_EMMISION_RATE_EXIT5, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT5, h_EMMISION_RATE_EXIT5,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT5(){
    return &h_env_EMMISION_RATE_EXIT5;
}



//constant setter
void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE_EXIT6){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT6, h_EMMISION_RATE_EXIT6, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT6, h_EMMISION_RATE_EXIT6,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT6(){
    return &h_env_EMMISION_RATE_EXIT6;
}



//constant setter
void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE_EXIT7){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT7, h_EMMISION_RATE_EXIT7, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT7, h_EMMISION_RATE_EXIT7,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT7(){
    return &h_env_EMMISION_RATE_EXIT7;
}



//constant setter
void set_EXIT1_PROBABILITY(int* h_EXIT1_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT1_PROBABILITY, h_EXIT1_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT1_PROBABILITY, h_EXIT1_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT1_PROBABILITY(){
    return &h_env_EXIT1_PROBABILITY;
}



//constant setter
void set_EXIT2_PROBABILITY(int* h_EXIT2_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT2_PROBABILITY, h_EXIT2_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT2_PROBABILITY, h_EXIT2_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT2_PROBABILITY(){
    return &h_env_EXIT2_PROBABILITY;
}



//constant setter
void set_EXIT3_PROBABILITY(int* h_EXIT3_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT3_PROBABILITY, h_EXIT3_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT3_PROBABILITY, h_EXIT3_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT3_PROBABILITY(){
    return &h_env_EXIT3_PROBABILITY;
}



//constant setter
void set_EXIT4_PROBABILITY(int* h_EXIT4_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT4_PROBABILITY, h_EXIT4_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT4_PROBABILITY, h_EXIT4_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT4_PROBABILITY(){
    return &h_env_EXIT4_PROBABILITY;
}



//constant setter
void set_EXIT5_PROBABILITY(int* h_EXIT5_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT5_PROBABILITY, h_EXIT5_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT5_PROBABILITY, h_EXIT5_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT5_PROBABILITY(){
    return &h_env_EXIT5_PROBABILITY;
}



//constant setter
void set_EXIT6_PROBABILITY(int* h_EXIT6_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT6_PROBABILITY, h_EXIT6_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT6_PROBABILITY, h_EXIT6_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT6_PROBABILITY(){
    return &h_env_EXIT6_PROBABILITY;
}



//constant setter
void set_EXIT7_PROBABILITY(int* h_EXIT7_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT7_PROBABILITY, h_EXIT7_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT7_PROBABILITY, h_EXIT7_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT7_PROBABILITY(){
    return &h_env_EXIT7_PROBABILITY;
}



//constant setter
void set_EXIT1_STATE(int* h_EXIT1_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT1_STATE, h_EXIT1_STATE, sizeof(int)));
    memcpy(&h_env_EXIT1_STATE, h_EXIT1_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT1_STATE(){
    return &h_env_EXIT1_STATE;
}



//constant setter
void set_EXIT2_STATE(int* h_EXIT2_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT2_STATE, h_EXIT2_STATE, sizeof(int)));
    memcpy(&h_env_EXIT2_STATE, h_EXIT2_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT2_STATE(){
    return &h_env_EXIT2_STATE;
}



//constant setter
void set_EXIT3_STATE(int* h_EXIT3_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT3_STATE, h_EXIT3_STATE, sizeof(int)));
    memcpy(&h_env_EXIT3_STATE, h_EXIT3_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT3_STATE(){
    return &h_env_EXIT3_STATE;
}



//constant setter
void set_EXIT4_STATE(int* h_EXIT4_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT4_STATE, h_EXIT4_STATE, sizeof(int)));
    memcpy(&h_env_EXIT4_STATE, h_EXIT4_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT4_STATE(){
    return &h_env_EXIT4_STATE;
}



//constant setter
void set_EXIT5_STATE(int* h_EXIT5_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT5_STATE, h_EXIT5_STATE, sizeof(int)));
    memcpy(&h_env_EXIT5_STATE, h_EXIT5_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT5_STATE(){
    return &h_env_EXIT5_STATE;
}



//constant setter
void set_EXIT6_STATE(int* h_EXIT6_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT6_STATE, h_EXIT6_STATE, sizeof(int)));
    memcpy(&h_env_EXIT6_STATE, h_EXIT6_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT6_STATE(){
    return &h_env_EXIT6_STATE;
}



//constant setter
void set_EXIT7_STATE(int* h_EXIT7_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT7_STATE, h_EXIT7_STATE, sizeof(int)));
    memcpy(&h_env_EXIT7_STATE, h_EXIT7_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT7_STATE(){
    return &h_env_EXIT7_STATE;
}



//constant setter
void set_EXIT1_CELL_COUNT(int* h_EXIT1_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT1_CELL_COUNT, h_EXIT1_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT1_CELL_COUNT, h_EXIT1_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT1_CELL_COUNT(){
    return &h_env_EXIT1_CELL_COUNT;
}



//constant setter
void set_EXIT2_CELL_COUNT(int* h_EXIT2_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT2_CELL_COUNT, h_EXIT2_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT2_CELL_COUNT, h_EXIT2_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT2_CELL_COUNT(){
    return &h_env_EXIT2_CELL_COUNT;
}



//constant setter
void set_EXIT3_CELL_COUNT(int* h_EXIT3_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT3_CELL_COUNT, h_EXIT3_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT3_CELL_COUNT, h_EXIT3_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT3_CELL_COUNT(){
    return &h_env_EXIT3_CELL_COUNT;
}



//constant setter
void set_EXIT4_CELL_COUNT(int* h_EXIT4_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT4_CELL_COUNT, h_EXIT4_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT4_CELL_COUNT, h_EXIT4_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT4_CELL_COUNT(){
    return &h_env_EXIT4_CELL_COUNT;
}



//constant setter
void set_EXIT5_CELL_COUNT(int* h_EXIT5_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT5_CELL_COUNT, h_EXIT5_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT5_CELL_COUNT, h_EXIT5_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT5_CELL_COUNT(){
    return &h_env_EXIT5_CELL_COUNT;
}



//constant setter
void set_EXIT6_CELL_COUNT(int* h_EXIT6_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT6_CELL_COUNT, h_EXIT6_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT6_CELL_COUNT, h_EXIT6_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT6_CELL_COUNT(){
    return &h_env_EXIT6_CELL_COUNT;
}



//constant setter
void set_EXIT7_CELL_COUNT(int* h_EXIT7_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT7_CELL_COUNT, h_EXIT7_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT7_CELL_COUNT, h_EXIT7_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT7_CELL_COUNT(){
    return &h_env_EXIT7_CELL_COUNT;
}



//constant setter
void set_TIME_SCALER(float* h_TIME_SCALER){
    gpuErrchk(cudaMemcpyToSymbol(TIME_SCALER, h_TIME_SCALER, sizeof(float)));
    memcpy(&h_env_TIME_SCALER, h_TIME_SCALER,sizeof(float));
}

//constant getter
const float* get_TIME_SCALER(){
    return &h_env_TIME_SCALER;
}



//constant setter
void set_STEER_WEIGHT(float* h_STEER_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(STEER_WEIGHT, h_STEER_WEIGHT, sizeof(float)));
    memcpy(&h_env_STEER_WEIGHT, h_STEER_WEIGHT,sizeof(float));
}

//constant getter
const float* get_STEER_WEIGHT(){
    return &h_env_STEER_WEIGHT;
}



//constant setter
void set_AVOID_WEIGHT(float* h_AVOID_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(AVOID_WEIGHT, h_AVOID_WEIGHT, sizeof(float)));
    memcpy(&h_env_AVOID_WEIGHT, h_AVOID_WEIGHT,sizeof(float));
}

//constant getter
const float* get_AVOID_WEIGHT(){
    return &h_env_AVOID_WEIGHT;
}



//constant setter
void set_COLLISION_WEIGHT(float* h_COLLISION_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(COLLISION_WEIGHT, h_COLLISION_WEIGHT, sizeof(float)));
    memcpy(&h_env_COLLISION_WEIGHT, h_COLLISION_WEIGHT,sizeof(float));
}

//constant getter
const float* get_COLLISION_WEIGHT(){
    return &h_env_COLLISION_WEIGHT;
}



//constant setter
void set_GOAL_WEIGHT(float* h_GOAL_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(GOAL_WEIGHT, h_GOAL_WEIGHT, sizeof(float)));
    memcpy(&h_env_GOAL_WEIGHT, h_GOAL_WEIGHT,sizeof(float));
}

//constant getter
const float* get_GOAL_WEIGHT(){
    return &h_env_GOAL_WEIGHT;
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

    
int get_agent_agent_MAX_count(){
    return xmachine_memory_agent_MAX;
}


int get_agent_agent_default_count(){
	//continuous agent
	return h_xmachine_memory_agent_default_count;
	
}

xmachine_memory_agent_list* get_device_agent_default_agents(){
	return d_agents_default;
}

xmachine_memory_agent_list* get_host_agent_default_agents(){
	return h_agents_default;
}

    
int get_agent_navmap_MAX_count(){
    return xmachine_memory_navmap_MAX;
}


int get_agent_navmap_static_count(){
	//discrete agent 
	return xmachine_memory_navmap_MAX;
}

xmachine_memory_navmap_list* get_device_navmap_static_agents(){
	return d_navmaps_static;
}

xmachine_memory_navmap_list* get_host_navmap_static_agents(){
	return h_navmaps_static;
}

int get_navmap_population_width(){
  return h_xmachine_memory_navmap_pop_width;
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

/** float get_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_agent_default_variable_x(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->x,
                    d_agents_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_agent_default_variable_y(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->y,
                    d_agents_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_velx(unsigned int index)
 * Gets the value of the velx variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable velx
 */
__host__ float get_agent_default_variable_velx(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_velx_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->velx,
                    d_agents_default->velx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_velx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->velx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access velx for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_vely(unsigned int index)
 * Gets the value of the vely variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable vely
 */
__host__ float get_agent_default_variable_vely(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_vely_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->vely,
                    d_agents_default->vely,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_vely_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->vely[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access vely for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_steer_x(unsigned int index)
 * Gets the value of the steer_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_x
 */
__host__ float get_agent_default_variable_steer_x(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_steer_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->steer_x,
                    d_agents_default->steer_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_steer_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->steer_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access steer_x for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_steer_y(unsigned int index)
 * Gets the value of the steer_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_y
 */
__host__ float get_agent_default_variable_steer_y(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_steer_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->steer_y,
                    d_agents_default->steer_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_steer_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->steer_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access steer_y for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_height(unsigned int index)
 * Gets the value of the height variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_agent_default_variable_height(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_height_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->height,
                    d_agents_default->height,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_height_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->height[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access height for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_agent_default_variable_exit_no(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_exit_no_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->exit_no,
                    d_agents_default->exit_no,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_exit_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->exit_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit_no for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_agent_default_variable_speed(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_speed_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->speed,
                    d_agents_default->speed,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_speed_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->speed[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access speed for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_agent_default_variable_lod(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_lod_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->lod,
                    d_agents_default->lod,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_lod_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->lod[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lod for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_agent_default_variable_animate(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_animate_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->animate,
                    d_agents_default->animate,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_animate_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->animate[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access animate for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_agent_default_variable_animate_dir(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_animate_dir_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->animate_dir,
                    d_agents_default->animate_dir,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_animate_dir_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->animate_dir[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access animate_dir for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_navmap_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_navmap_static_variable_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->x,
                    d_navmaps_static->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_navmap_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_navmap_static_variable_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->y,
                    d_navmaps_static->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_navmap_static_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_navmap_static_variable_exit_no(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit_no_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit_no,
                    d_navmaps_static->exit_no,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit_no for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_height(unsigned int index)
 * Gets the value of the height variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_navmap_static_variable_height(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_height_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->height,
                    d_navmaps_static->height,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_height_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->height[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access height for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_collision_x(unsigned int index)
 * Gets the value of the collision_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_x
 */
__host__ float get_navmap_static_variable_collision_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_collision_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->collision_x,
                    d_navmaps_static->collision_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_collision_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->collision_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access collision_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_collision_y(unsigned int index)
 * Gets the value of the collision_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_y
 */
__host__ float get_navmap_static_variable_collision_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_collision_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->collision_y,
                    d_navmaps_static->collision_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_collision_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->collision_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access collision_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit0_x(unsigned int index)
 * Gets the value of the exit0_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_x
 */
__host__ float get_navmap_static_variable_exit0_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit0_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit0_x,
                    d_navmaps_static->exit0_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit0_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit0_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit0_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit0_y(unsigned int index)
 * Gets the value of the exit0_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_y
 */
__host__ float get_navmap_static_variable_exit0_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit0_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit0_y,
                    d_navmaps_static->exit0_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit0_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit0_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit0_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit1_x(unsigned int index)
 * Gets the value of the exit1_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_x
 */
__host__ float get_navmap_static_variable_exit1_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit1_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit1_x,
                    d_navmaps_static->exit1_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit1_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit1_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit1_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit1_y(unsigned int index)
 * Gets the value of the exit1_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_y
 */
__host__ float get_navmap_static_variable_exit1_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit1_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit1_y,
                    d_navmaps_static->exit1_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit1_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit1_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit1_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit2_x(unsigned int index)
 * Gets the value of the exit2_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_x
 */
__host__ float get_navmap_static_variable_exit2_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit2_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit2_x,
                    d_navmaps_static->exit2_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit2_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit2_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit2_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit2_y(unsigned int index)
 * Gets the value of the exit2_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_y
 */
__host__ float get_navmap_static_variable_exit2_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit2_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit2_y,
                    d_navmaps_static->exit2_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit2_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit2_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit2_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit3_x(unsigned int index)
 * Gets the value of the exit3_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_x
 */
__host__ float get_navmap_static_variable_exit3_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit3_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit3_x,
                    d_navmaps_static->exit3_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit3_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit3_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit3_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit3_y(unsigned int index)
 * Gets the value of the exit3_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_y
 */
__host__ float get_navmap_static_variable_exit3_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit3_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit3_y,
                    d_navmaps_static->exit3_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit3_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit3_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit3_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit4_x(unsigned int index)
 * Gets the value of the exit4_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_x
 */
__host__ float get_navmap_static_variable_exit4_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit4_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit4_x,
                    d_navmaps_static->exit4_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit4_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit4_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit4_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit4_y(unsigned int index)
 * Gets the value of the exit4_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_y
 */
__host__ float get_navmap_static_variable_exit4_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit4_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit4_y,
                    d_navmaps_static->exit4_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit4_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit4_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit4_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit5_x(unsigned int index)
 * Gets the value of the exit5_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_x
 */
__host__ float get_navmap_static_variable_exit5_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit5_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit5_x,
                    d_navmaps_static->exit5_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit5_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit5_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit5_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit5_y(unsigned int index)
 * Gets the value of the exit5_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_y
 */
__host__ float get_navmap_static_variable_exit5_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit5_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit5_y,
                    d_navmaps_static->exit5_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit5_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit5_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit5_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit6_x(unsigned int index)
 * Gets the value of the exit6_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_x
 */
__host__ float get_navmap_static_variable_exit6_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit6_x_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit6_x,
                    d_navmaps_static->exit6_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit6_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit6_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit6_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit6_y(unsigned int index)
 * Gets the value of the exit6_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_y
 */
__host__ float get_navmap_static_variable_exit6_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit6_y_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit6_y,
                    d_navmaps_static->exit6_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit6_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit6_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit6_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_agent_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_agent_hostToDevice(xmachine_memory_agent_list * d_dst, xmachine_memory_agent * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->velx, &h_agent->velx, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->vely, &h_agent->vely, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_x, &h_agent->steer_x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_y, &h_agent->steer_y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->height, &h_agent->height, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->exit_no, &h_agent->exit_no, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->speed, &h_agent->speed, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lod, &h_agent->lod, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate, &h_agent->animate, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate_dir, &h_agent->animate_dir, sizeof(int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @todo - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_agent_hostToDevice(xmachine_memory_agent_list * d_dst, xmachine_memory_agent_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->velx, h_src->velx, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->vely, h_src->vely, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_x, h_src->steer_x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_y, h_src->steer_y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->height, h_src->height, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->exit_no, h_src->exit_no, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->speed, h_src->speed, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lod, h_src->lod, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate, h_src->animate, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate_dir, h_src->animate_dir, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_agent* h_allocate_agent_agent(){
	xmachine_memory_agent* agent = (xmachine_memory_agent*)malloc(sizeof(xmachine_memory_agent));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_agent));

	return agent;
}
void h_free_agent_agent(xmachine_memory_agent** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_agent** h_allocate_agent_agent_array(unsigned int count){
	xmachine_memory_agent ** agents = (xmachine_memory_agent**)malloc(count * sizeof(xmachine_memory_agent*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_agent();
	}
	return agents;
}
void h_free_agent_agent_array(xmachine_memory_agent*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_agent(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_agent_AoS_to_SoA(xmachine_memory_agent_list * dst, xmachine_memory_agent** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->velx[i] = src[i]->velx;
			 
			dst->vely[i] = src[i]->vely;
			 
			dst->steer_x[i] = src[i]->steer_x;
			 
			dst->steer_y[i] = src[i]->steer_y;
			 
			dst->height[i] = src[i]->height;
			 
			dst->exit_no[i] = src[i]->exit_no;
			 
			dst->speed[i] = src[i]->speed;
			 
			dst->lod[i] = src[i]->lod;
			 
			dst->animate[i] = src[i]->animate;
			 
			dst->animate_dir[i] = src[i]->animate_dir;
			
		}
	}
}


void h_add_agent_agent_default(xmachine_memory_agent* agent){
	if (h_xmachine_memory_agent_count + 1 > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of agent agents in state default will be exceeded by h_add_agent_agent_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_agent_hostToDevice(d_agents_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_agent_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_agents_default, d_agents_new, h_xmachine_memory_agent_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_agent_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_agents_default_variable_x_data_iteration = 0;
    h_agents_default_variable_y_data_iteration = 0;
    h_agents_default_variable_velx_data_iteration = 0;
    h_agents_default_variable_vely_data_iteration = 0;
    h_agents_default_variable_steer_x_data_iteration = 0;
    h_agents_default_variable_steer_y_data_iteration = 0;
    h_agents_default_variable_height_data_iteration = 0;
    h_agents_default_variable_exit_no_data_iteration = 0;
    h_agents_default_variable_speed_data_iteration = 0;
    h_agents_default_variable_lod_data_iteration = 0;
    h_agents_default_variable_animate_data_iteration = 0;
    h_agents_default_variable_animate_dir_data_iteration = 0;
    

}
void h_add_agents_agent_default(xmachine_memory_agent** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_agent_count + count > xmachine_memory_agent_MAX){
			printf("Error: Buffer size of agent agents in state default will be exceeded by h_add_agents_agent_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_agent_AoS_to_SoA(h_agents_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_agent_hostToDevice(d_agents_new, h_agents_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_agent_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_agents_default, d_agents_new, h_xmachine_memory_agent_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_agent_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_agents_default_variable_x_data_iteration = 0;
        h_agents_default_variable_y_data_iteration = 0;
        h_agents_default_variable_velx_data_iteration = 0;
        h_agents_default_variable_vely_data_iteration = 0;
        h_agents_default_variable_steer_x_data_iteration = 0;
        h_agents_default_variable_steer_y_data_iteration = 0;
        h_agents_default_variable_height_data_iteration = 0;
        h_agents_default_variable_exit_no_data_iteration = 0;
        h_agents_default_variable_speed_data_iteration = 0;
        h_agents_default_variable_lod_data_iteration = 0;
        h_agents_default_variable_animate_data_iteration = 0;
        h_agents_default_variable_animate_dir_data_iteration = 0;
        

	}
}


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
float reduce_agent_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->x),  thrust::device_pointer_cast(d_agents_default->x) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->y),  thrust::device_pointer_cast(d_agents_default->y) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_velx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->velx),  thrust::device_pointer_cast(d_agents_default->velx) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_velx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->velx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_velx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->velx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_vely_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->vely),  thrust::device_pointer_cast(d_agents_default->vely) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_vely_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->vely);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_vely_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->vely);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_steer_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->steer_x),  thrust::device_pointer_cast(d_agents_default->steer_x) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_steer_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_steer_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_steer_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->steer_y),  thrust::device_pointer_cast(d_agents_default->steer_y) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_steer_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_steer_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_height_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->height),  thrust::device_pointer_cast(d_agents_default->height) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_height_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->height);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_height_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->height);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_exit_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->exit_no),  thrust::device_pointer_cast(d_agents_default->exit_no) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_exit_no_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->exit_no),  thrust::device_pointer_cast(d_agents_default->exit_no) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_exit_no_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->exit_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_exit_no_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->exit_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_speed_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->speed),  thrust::device_pointer_cast(d_agents_default->speed) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_speed_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->speed);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_speed_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->speed);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_lod_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->lod),  thrust::device_pointer_cast(d_agents_default->lod) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_lod_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->lod),  thrust::device_pointer_cast(d_agents_default->lod) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_lod_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->lod);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_lod_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->lod);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_animate_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->animate),  thrust::device_pointer_cast(d_agents_default->animate) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_animate_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_animate_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_animate_dir_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->animate_dir),  thrust::device_pointer_cast(d_agents_default->animate_dir) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_animate_dir_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->animate_dir),  thrust::device_pointer_cast(d_agents_default->animate_dir) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_animate_dir_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate_dir);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_animate_dir_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate_dir);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_navmap_static_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->x),  thrust::device_pointer_cast(d_navmaps_static->x) + h_xmachine_memory_navmap_static_count);
}

int count_navmap_static_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->x),  thrust::device_pointer_cast(d_navmaps_static->x) + h_xmachine_memory_navmap_static_count, count_value);
}
int min_navmap_static_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_navmap_static_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_navmap_static_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->y),  thrust::device_pointer_cast(d_navmaps_static->y) + h_xmachine_memory_navmap_static_count);
}

int count_navmap_static_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->y),  thrust::device_pointer_cast(d_navmaps_static->y) + h_xmachine_memory_navmap_static_count, count_value);
}
int min_navmap_static_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_navmap_static_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_navmap_static_exit_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit_no),  thrust::device_pointer_cast(d_navmaps_static->exit_no) + h_xmachine_memory_navmap_static_count);
}

int count_navmap_static_exit_no_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->exit_no),  thrust::device_pointer_cast(d_navmaps_static->exit_no) + h_xmachine_memory_navmap_static_count, count_value);
}
int min_navmap_static_exit_no_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_navmap_static_exit_no_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_height_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->height),  thrust::device_pointer_cast(d_navmaps_static->height) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_height_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->height);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_height_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->height);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_collision_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->collision_x),  thrust::device_pointer_cast(d_navmaps_static->collision_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_collision_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_collision_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_collision_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->collision_y),  thrust::device_pointer_cast(d_navmaps_static->collision_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_collision_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_collision_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit0_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit0_x),  thrust::device_pointer_cast(d_navmaps_static->exit0_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit0_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit0_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit0_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit0_y),  thrust::device_pointer_cast(d_navmaps_static->exit0_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit0_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit0_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit1_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit1_x),  thrust::device_pointer_cast(d_navmaps_static->exit1_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit1_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit1_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit1_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit1_y),  thrust::device_pointer_cast(d_navmaps_static->exit1_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit1_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit1_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit2_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit2_x),  thrust::device_pointer_cast(d_navmaps_static->exit2_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit2_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit2_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit2_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit2_y),  thrust::device_pointer_cast(d_navmaps_static->exit2_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit2_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit2_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit3_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit3_x),  thrust::device_pointer_cast(d_navmaps_static->exit3_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit3_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit3_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit3_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit3_y),  thrust::device_pointer_cast(d_navmaps_static->exit3_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit3_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit3_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit4_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit4_x),  thrust::device_pointer_cast(d_navmaps_static->exit4_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit4_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit4_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit4_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit4_y),  thrust::device_pointer_cast(d_navmaps_static->exit4_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit4_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit4_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit5_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit5_x),  thrust::device_pointer_cast(d_navmaps_static->exit5_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit5_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit5_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit5_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit5_y),  thrust::device_pointer_cast(d_navmaps_static->exit5_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit5_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit5_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit6_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit6_x),  thrust::device_pointer_cast(d_navmaps_static->exit6_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit6_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit6_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit6_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit6_y),  thrust::device_pointer_cast(d_navmaps_static->exit6_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit6_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit6_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
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



	
/* Shared memory size calculator for agent function */
int agent_output_pedestrian_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_pedestrian_location
 * Agent function prototype for output_pedestrian_location function of agent agent
 */
void agent_output_pedestrian_location(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_pedestrian_location_count + h_xmachine_memory_agent_count > xmachine_message_pedestrian_location_MAX){
		printf("Error: Buffer size of pedestrian_location message will be exceeded in function output_pedestrian_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_pedestrian_location, agent_output_pedestrian_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_pedestrian_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_pedestrian_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_pedestrian_location_output_type, &h_message_pedestrian_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_pedestrian_location)
	//Reallocate   : false
	//Input        : 
	//Output       : pedestrian_location
	//Agent Output : 
	GPUFLAME_output_pedestrian_location<<<g, b, sm_size, stream>>>(d_agents, d_pedestrian_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_pedestrian_location_count += h_xmachine_memory_agent_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_pedestrian_location_count, &h_message_pedestrian_location_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_pedestrian_location_partition_matrix, 0, sizeof(xmachine_message_pedestrian_location_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_pedestrian_location_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_pedestrian_location_messages, no_sm, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize;
	  hist_pedestrian_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_location_local_bin_index, d_xmachine_message_pedestrian_location_unsorted_index, d_pedestrian_location_partition_matrix->end_or_count, d_pedestrian_locations, h_message_pedestrian_location_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_scan_tmp_memory_pedestrian_location, 
          scan_tmp_bytes_pedestrian_location, 
          d_pedestrian_location_partition_matrix->end_or_count,
          d_pedestrian_location_partition_matrix->start,
          xmachine_message_pedestrian_location_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_pedestrian_location_messages, no_sm, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_pedestrian_location_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_location_local_bin_index, d_xmachine_message_pedestrian_location_unsorted_index, d_pedestrian_location_partition_matrix->start, d_pedestrian_locations, d_pedestrian_locations_swap, h_message_pedestrian_location_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_pedestrian_location_messages, no_sm, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize;
	  hash_pedestrian_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_location_keys, d_xmachine_message_pedestrian_location_values, d_pedestrian_locations);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_pedestrian_location_keys),  thrust::device_pointer_cast(d_xmachine_message_pedestrian_location_keys) + h_message_pedestrian_location_count,  thrust::device_pointer_cast(d_xmachine_message_pedestrian_location_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_pedestrian_location_partition_matrix->start, 0xffffffff, xmachine_message_pedestrian_location_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_pedestrian_location_messages, reorder_messages_sm_size, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_pedestrian_location_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_pedestrian_location_keys, d_xmachine_message_pedestrian_location_values, d_pedestrian_location_partition_matrix, d_pedestrian_locations, d_pedestrian_locations_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_pedestrian_location_list* d_pedestrian_locations_temp = d_pedestrian_locations;
	d_pedestrian_locations = d_pedestrian_locations_swap;
	d_pedestrian_locations_swap = d_pedestrian_locations_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_pedestrian_location agents in state default will be exceeded moving working agents to next state in function output_pedestrian_location\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_avoid_pedestrians_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_pedestrian_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_avoid_pedestrians
 * Agent function prototype for avoid_pedestrians function of agent agent
 */
void agent_avoid_pedestrians(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_avoid_pedestrians, agent_avoid_pedestrians_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_avoid_pedestrians_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_pedestrian_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_x_byte_offset, tex_xmachine_message_pedestrian_location_x, d_pedestrian_locations->x, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_x_offset = (int)tex_xmachine_message_pedestrian_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_x_offset, &h_tex_xmachine_message_pedestrian_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_y_byte_offset, tex_xmachine_message_pedestrian_location_y, d_pedestrian_locations->y, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_y_offset = (int)tex_xmachine_message_pedestrian_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_y_offset, &h_tex_xmachine_message_pedestrian_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_z_byte_offset, tex_xmachine_message_pedestrian_location_z, d_pedestrian_locations->z, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_z_offset = (int)tex_xmachine_message_pedestrian_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_z_offset, &h_tex_xmachine_message_pedestrian_location_z_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_pedestrian_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_pbm_start_byte_offset, tex_xmachine_message_pedestrian_location_pbm_start, d_pedestrian_location_partition_matrix->start, sizeof(int)*xmachine_message_pedestrian_location_grid_size));
	h_tex_xmachine_message_pedestrian_location_pbm_start_offset = (int)tex_xmachine_message_pedestrian_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_pbm_start_offset, &h_tex_xmachine_message_pedestrian_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset, tex_xmachine_message_pedestrian_location_pbm_end_or_count, d_pedestrian_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_pedestrian_location_grid_size));
  h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset = (int)tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset, &h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (avoid_pedestrians)
	//Reallocate   : false
	//Input        : pedestrian_location
	//Output       : 
	//Agent Output : 
	GPUFLAME_avoid_pedestrians<<<g, b, sm_size, stream>>>(d_agents, d_pedestrian_locations, d_pedestrian_location_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_z));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of avoid_pedestrians agents in state default will be exceeded moving working agents to next state in function avoid_pedestrians\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_force_flow_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has discrete partitioning
	//Will be reading using texture lookups so sm size can stay the same but need to hold range and width
	sm_size += (blockSize * sizeof(xmachine_message_navmap_cell));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_force_flow
 * Agent function prototype for force_flow function of agent agent
 */
void agent_force_flow(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_force_flow, agent_force_flow_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_force_flow_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_navmap_cell_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_x_byte_offset, tex_xmachine_message_navmap_cell_x, d_navmap_cells->x, sizeof(int)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_x_offset = (int)tex_xmachine_message_navmap_cell_x_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_x_offset, &h_tex_xmachine_message_navmap_cell_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_y_byte_offset, tex_xmachine_message_navmap_cell_y, d_navmap_cells->y, sizeof(int)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_y_offset = (int)tex_xmachine_message_navmap_cell_y_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_y_offset, &h_tex_xmachine_message_navmap_cell_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit_no_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit_no_byte_offset, tex_xmachine_message_navmap_cell_exit_no, d_navmap_cells->exit_no, sizeof(int)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit_no_offset = (int)tex_xmachine_message_navmap_cell_exit_no_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit_no_offset, &h_tex_xmachine_message_navmap_cell_exit_no_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_height_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_height_byte_offset, tex_xmachine_message_navmap_cell_height, d_navmap_cells->height, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_height_offset = (int)tex_xmachine_message_navmap_cell_height_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_height_offset, &h_tex_xmachine_message_navmap_cell_height_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_collision_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_collision_x_byte_offset, tex_xmachine_message_navmap_cell_collision_x, d_navmap_cells->collision_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_collision_x_offset = (int)tex_xmachine_message_navmap_cell_collision_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_collision_x_offset, &h_tex_xmachine_message_navmap_cell_collision_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_collision_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_collision_y_byte_offset, tex_xmachine_message_navmap_cell_collision_y, d_navmap_cells->collision_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_collision_y_offset = (int)tex_xmachine_message_navmap_cell_collision_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_collision_y_offset, &h_tex_xmachine_message_navmap_cell_collision_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit0_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit0_x_byte_offset, tex_xmachine_message_navmap_cell_exit0_x, d_navmap_cells->exit0_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit0_x_offset = (int)tex_xmachine_message_navmap_cell_exit0_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit0_x_offset, &h_tex_xmachine_message_navmap_cell_exit0_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit0_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit0_y_byte_offset, tex_xmachine_message_navmap_cell_exit0_y, d_navmap_cells->exit0_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit0_y_offset = (int)tex_xmachine_message_navmap_cell_exit0_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit0_y_offset, &h_tex_xmachine_message_navmap_cell_exit0_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit1_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit1_x_byte_offset, tex_xmachine_message_navmap_cell_exit1_x, d_navmap_cells->exit1_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit1_x_offset = (int)tex_xmachine_message_navmap_cell_exit1_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit1_x_offset, &h_tex_xmachine_message_navmap_cell_exit1_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit1_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit1_y_byte_offset, tex_xmachine_message_navmap_cell_exit1_y, d_navmap_cells->exit1_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit1_y_offset = (int)tex_xmachine_message_navmap_cell_exit1_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit1_y_offset, &h_tex_xmachine_message_navmap_cell_exit1_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit2_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit2_x_byte_offset, tex_xmachine_message_navmap_cell_exit2_x, d_navmap_cells->exit2_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit2_x_offset = (int)tex_xmachine_message_navmap_cell_exit2_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit2_x_offset, &h_tex_xmachine_message_navmap_cell_exit2_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit2_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit2_y_byte_offset, tex_xmachine_message_navmap_cell_exit2_y, d_navmap_cells->exit2_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit2_y_offset = (int)tex_xmachine_message_navmap_cell_exit2_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit2_y_offset, &h_tex_xmachine_message_navmap_cell_exit2_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit3_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit3_x_byte_offset, tex_xmachine_message_navmap_cell_exit3_x, d_navmap_cells->exit3_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit3_x_offset = (int)tex_xmachine_message_navmap_cell_exit3_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit3_x_offset, &h_tex_xmachine_message_navmap_cell_exit3_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit3_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit3_y_byte_offset, tex_xmachine_message_navmap_cell_exit3_y, d_navmap_cells->exit3_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit3_y_offset = (int)tex_xmachine_message_navmap_cell_exit3_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit3_y_offset, &h_tex_xmachine_message_navmap_cell_exit3_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit4_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit4_x_byte_offset, tex_xmachine_message_navmap_cell_exit4_x, d_navmap_cells->exit4_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit4_x_offset = (int)tex_xmachine_message_navmap_cell_exit4_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit4_x_offset, &h_tex_xmachine_message_navmap_cell_exit4_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit4_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit4_y_byte_offset, tex_xmachine_message_navmap_cell_exit4_y, d_navmap_cells->exit4_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit4_y_offset = (int)tex_xmachine_message_navmap_cell_exit4_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit4_y_offset, &h_tex_xmachine_message_navmap_cell_exit4_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit5_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit5_x_byte_offset, tex_xmachine_message_navmap_cell_exit5_x, d_navmap_cells->exit5_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit5_x_offset = (int)tex_xmachine_message_navmap_cell_exit5_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit5_x_offset, &h_tex_xmachine_message_navmap_cell_exit5_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit5_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit5_y_byte_offset, tex_xmachine_message_navmap_cell_exit5_y, d_navmap_cells->exit5_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit5_y_offset = (int)tex_xmachine_message_navmap_cell_exit5_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit5_y_offset, &h_tex_xmachine_message_navmap_cell_exit5_y_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit6_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit6_x_byte_offset, tex_xmachine_message_navmap_cell_exit6_x, d_navmap_cells->exit6_x, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit6_x_offset = (int)tex_xmachine_message_navmap_cell_exit6_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit6_x_offset, &h_tex_xmachine_message_navmap_cell_exit6_x_offset, sizeof(int)));
	size_t tex_xmachine_message_navmap_cell_exit6_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_navmap_cell_exit6_y_byte_offset, tex_xmachine_message_navmap_cell_exit6_y, d_navmap_cells->exit6_y, sizeof(float)*xmachine_message_navmap_cell_MAX));
	h_tex_xmachine_message_navmap_cell_exit6_y_offset = (int)tex_xmachine_message_navmap_cell_exit6_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_navmap_cell_exit6_y_offset, &h_tex_xmachine_message_navmap_cell_exit6_y_offset, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (force_flow)
	//Reallocate   : true
	//Input        : navmap_cell
	//Output       : 
	//Agent Output : 
	GPUFLAME_force_flow<<<g, b, sm_size, stream>>>(d_agents, d_navmap_cells, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit_no));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_height));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_collision_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_collision_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit0_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit0_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit1_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit1_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit2_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit2_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit3_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit3_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit4_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit4_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit5_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit5_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit6_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_navmap_cell_exit6_y));
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents->_scan_input,
        d_agents->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_agent_list* force_flow_agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = force_flow_agents_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_agent_count = scan_last_sum+1;
	else
		h_xmachine_memory_agent_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of force_flow agents in state default will be exceeded moving working agents to next state in function force_flow\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents, h_xmachine_memory_agent_default_count, h_xmachine_memory_agent_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_move
 * Agent function prototype for move function of agent agent
 */
void agent_move(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_move, agent_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_move_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (move)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_move<<<g, b, sm_size, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of move agents in state default will be exceeded moving working agents to next state in function move\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int navmap_output_navmap_cells_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** navmap_output_navmap_cells
 * Agent function prototype for output_navmap_cells function of navmap agent
 */
void navmap_output_navmap_cells(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_navmap_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_navmap_static_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_navmap_list* navmaps_static_temp = d_navmaps;
	d_navmaps = d_navmaps_static;
	d_navmaps_static = navmaps_static_temp;
	//set working count to current state count
	h_xmachine_memory_navmap_count = h_xmachine_memory_navmap_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_count, &h_xmachine_memory_navmap_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_navmap_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_navmap_cells, navmap_output_navmap_cells_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = navmap_output_navmap_cells_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_navmap_cells)
	//Reallocate   : false
	//Input        : 
	//Output       : navmap_cell
	//Agent Output : 
	GPUFLAME_output_navmap_cells<<<g, b, sm_size, stream>>>(d_navmaps, d_navmap_cells);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	navmaps_static_temp = d_navmaps_static;
	d_navmaps_static = d_navmaps;
	d_navmaps = navmaps_static_temp;
    //set current state count
	h_xmachine_memory_navmap_static_count = h_xmachine_memory_navmap_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int navmap_generate_pedestrians_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** navmap_generate_pedestrians
 * Agent function prototype for generate_pedestrians function of navmap agent
 */
void navmap_generate_pedestrians(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_navmap_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_navmap_static_count;

	
	//FOR agent AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_navmap_list* navmaps_static_temp = d_navmaps;
	d_navmaps = d_navmaps_static;
	d_navmaps_static = navmaps_static_temp;
	//set working count to current state count
	h_xmachine_memory_navmap_count = h_xmachine_memory_navmap_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_count, &h_xmachine_memory_navmap_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_navmap_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_pedestrians, navmap_generate_pedestrians_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = navmap_generate_pedestrians_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_pedestrians)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : agent
	GPUFLAME_generate_pedestrians<<<g, b, sm_size, stream>>>(d_navmaps, d_agents_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE navmap AGENTS ARE KILLED (needed for scatter)
	int navmaps_pre_death_count = h_xmachine_memory_navmap_count;
	
	//FOR agent AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents_new->_scan_input, 
        d_agents_new->_position, 
        navmaps_pre_death_count,
        stream
    );

	//reset agent count
	int agent_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents_new->_position[navmaps_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents_new->_scan_input[navmaps_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		agent_after_birth_count = h_xmachine_memory_agent_default_count + scan_last_sum+1;
	else
		agent_after_birth_count = h_xmachine_memory_agent_default_count + scan_last_sum;
	//check buffer is not exceeded
	if (agent_after_birth_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of agent agents in state default will be exceeded writing new agents in function generate_pedestrians\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents_new, h_xmachine_memory_agent_default_count, navmaps_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_agent_default_count = agent_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	navmaps_static_temp = d_navmaps_static;
	d_navmaps_static = d_navmaps;
	d_navmaps = navmaps_static_temp;
    //set current state count
	h_xmachine_memory_navmap_static_count = h_xmachine_memory_navmap_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
	
}


 
extern void reset_FloodCell_Default_count()
{
    h_xmachine_memory_FloodCell_Default_count = 0;
}
 
extern void reset_agent_default_count()
{
    h_xmachine_memory_agent_default_count = 0;
}
 
extern void reset_navmap_static_count()
{
    h_xmachine_memory_navmap_static_count = 0;
}