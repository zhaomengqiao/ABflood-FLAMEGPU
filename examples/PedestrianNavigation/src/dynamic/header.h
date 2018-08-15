
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



#ifndef __HEADER
#define __HEADER

#if defined __NVCC__
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__
#define __FLAME_GPU_HOST_FUNC__ __host__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

// FLAME GPU Version Macros.
#define FLAME_GPU_MAJOR_VERSION 1
#define FLAME_GPU_MINOR_VERSION 5
#define FLAME_GPU_PATCH_VERSION 0

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 65536

//Maximum population size of xmachine_memory_agent
#define xmachine_memory_agent_MAX 65536

//Maximum population size of xmachine_memory_navmap
#define xmachine_memory_navmap_MAX 65536


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_pedestrian_location
#define xmachine_message_pedestrian_location_MAX 65536

//Maximum population size of xmachine_mmessage_navmap_cell
#define xmachine_message_navmap_cell_MAX 65536


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_pedestrian_location_partitioningSpatial
#define xmachine_message_navmap_cell_partitioningDiscrete

/* Spatial partitioning grid size definitions */
//xmachine_message_pedestrian_location partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_pedestrian_location_grid_size 6400

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_agent
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float velx;    /**< X-machine memory variable velx of type float.*/
    float vely;    /**< X-machine memory variable vely of type float.*/
    float steer_x;    /**< X-machine memory variable steer_x of type float.*/
    float steer_y;    /**< X-machine memory variable steer_y of type float.*/
    float height;    /**< X-machine memory variable height of type float.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float speed;    /**< X-machine memory variable speed of type float.*/
    int lod;    /**< X-machine memory variable lod of type int.*/
    float animate;    /**< X-machine memory variable animate of type float.*/
    int animate_dir;    /**< X-machine memory variable animate_dir of type int.*/
};

/** struct xmachine_memory_navmap
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_navmap
{
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float height;    /**< X-machine memory variable height of type float.*/
    float collision_x;    /**< X-machine memory variable collision_x of type float.*/
    float collision_y;    /**< X-machine memory variable collision_y of type float.*/
    float exit0_x;    /**< X-machine memory variable exit0_x of type float.*/
    float exit0_y;    /**< X-machine memory variable exit0_y of type float.*/
    float exit1_x;    /**< X-machine memory variable exit1_x of type float.*/
    float exit1_y;    /**< X-machine memory variable exit1_y of type float.*/
    float exit2_x;    /**< X-machine memory variable exit2_x of type float.*/
    float exit2_y;    /**< X-machine memory variable exit2_y of type float.*/
    float exit3_x;    /**< X-machine memory variable exit3_x of type float.*/
    float exit3_y;    /**< X-machine memory variable exit3_y of type float.*/
    float exit4_x;    /**< X-machine memory variable exit4_x of type float.*/
    float exit4_y;    /**< X-machine memory variable exit4_y of type float.*/
    float exit5_x;    /**< X-machine memory variable exit5_x of type float.*/
    float exit5_y;    /**< X-machine memory variable exit5_y of type float.*/
    float exit6_x;    /**< X-machine memory variable exit6_x of type float.*/
    float exit6_y;    /**< X-machine memory variable exit6_y of type float.*/
};



/* Message structures */

/** struct xmachine_message_pedestrian_location
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_pedestrian_location
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/
};

/** struct xmachine_message_navmap_cell
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_navmap_cell
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    int exit_no;        /**< Message variable exit_no of type int.*/  
    float height;        /**< Message variable height of type float.*/  
    float collision_x;        /**< Message variable collision_x of type float.*/  
    float collision_y;        /**< Message variable collision_y of type float.*/  
    float exit0_x;        /**< Message variable exit0_x of type float.*/  
    float exit0_y;        /**< Message variable exit0_y of type float.*/  
    float exit1_x;        /**< Message variable exit1_x of type float.*/  
    float exit1_y;        /**< Message variable exit1_y of type float.*/  
    float exit2_x;        /**< Message variable exit2_x of type float.*/  
    float exit2_y;        /**< Message variable exit2_y of type float.*/  
    float exit3_x;        /**< Message variable exit3_x of type float.*/  
    float exit3_y;        /**< Message variable exit3_y of type float.*/  
    float exit4_x;        /**< Message variable exit4_x of type float.*/  
    float exit4_y;        /**< Message variable exit4_y of type float.*/  
    float exit5_x;        /**< Message variable exit5_x of type float.*/  
    float exit5_y;        /**< Message variable exit5_y of type float.*/  
    float exit6_x;        /**< Message variable exit6_x of type float.*/  
    float exit6_y;        /**< Message variable exit6_y of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list y of type float.*/
    float velx [xmachine_memory_agent_MAX];    /**< X-machine memory variable list velx of type float.*/
    float vely [xmachine_memory_agent_MAX];    /**< X-machine memory variable list vely of type float.*/
    float steer_x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_x of type float.*/
    float steer_y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_y of type float.*/
    float height [xmachine_memory_agent_MAX];    /**< X-machine memory variable list height of type float.*/
    int exit_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float speed [xmachine_memory_agent_MAX];    /**< X-machine memory variable list speed of type float.*/
    int lod [xmachine_memory_agent_MAX];    /**< X-machine memory variable list lod of type int.*/
    float animate [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate of type float.*/
    int animate_dir [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate_dir of type int.*/
};

/** struct xmachine_memory_navmap_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_navmap_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_navmap_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_navmap_MAX];  /**< Used during parallel prefix sum */
    
    int x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list y of type int.*/
    int exit_no [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float height [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list height of type float.*/
    float collision_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_x of type float.*/
    float collision_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_y of type float.*/
    float exit0_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_y of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_pedestrian_location_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_pedestrian_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_pedestrian_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_pedestrian_location_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list z of type float.*/
    
};

/** struct xmachine_message_navmap_cell_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_navmap_cell_list
{
    int x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list y of type int.*/
    int exit_no [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit_no of type int.*/
    float height [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list height of type float.*/
    float collision_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_x of type float.*/
    float collision_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_y of type float.*/
    float exit0_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit6_y of type float.*/
    
};



/* Spatially Partitioned Message boundary Matrices */

/** struct xmachine_message_pedestrian_location_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_pedestrian_location 
 */
struct xmachine_message_pedestrian_location_PBM
{
	int start[xmachine_message_pedestrian_location_grid_size];
	int end_or_count[xmachine_message_pedestrian_location_grid_size];
};



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * output_pedestrian_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages Pointer to output message list of type xmachine_message_pedestrian_location_list. Must be passed as an argument to the add_pedestrian_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages);

/**
 * avoid_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages  pedestrian_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pedestrian_location_message and get_next_pedestrian_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_pedestrian_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * force_flow FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages  navmap_cell_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_navmap_cell_message and get_next_navmap_cell_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int force_flow(xmachine_memory_agent* agent, xmachine_message_navmap_cell_list* navmap_cell_messages, RNG_rand48* rand48);

/**
 * move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent);

/**
 * output_navmap_cells FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages Pointer to output message list of type xmachine_message_navmap_cell_list. Must be passed as an argument to the add_navmap_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_navmap_cells(xmachine_memory_navmap* agent, xmachine_message_navmap_cell_list* navmap_cell_messages);

/**
 * generate_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48);

  
/* Message Function Prototypes for Spatially Partitioned pedestrian_location message implemented in FLAMEGPU_Kernels */

/** add_pedestrian_location_message
 * Function for all types of message partitioning
 * Adds a new pedestrian_location agent to the xmachine_memory_pedestrian_location_list list using a linear mapping
 * @param agents	xmachine_memory_pedestrian_location_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, float x, float y, float z);
 
/** get_first_pedestrian_location_message
 * Get first message function for spatially partitioned messages
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_first_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, float x, float y, float z);

/** get_next_pedestrian_location_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_next_pedestrian_location_message(xmachine_message_pedestrian_location* current, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix);

  
/* Message Function Prototypes for Discrete Partitioned navmap_cell message implemented in FLAMEGPU_Kernels */

/** add_navmap_cell_message
 * Function for all types of message partitioning
 * Adds a new navmap_cell agent to the xmachine_memory_navmap_cell_list list using a linear mapping
 * @param agents	xmachine_memory_navmap_cell_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param exit_no	message variable of type int
 * @param height	message variable of type float
 * @param collision_x	message variable of type float
 * @param collision_y	message variable of type float
 * @param exit0_x	message variable of type float
 * @param exit0_y	message variable of type float
 * @param exit1_x	message variable of type float
 * @param exit1_y	message variable of type float
 * @param exit2_x	message variable of type float
 * @param exit2_y	message variable of type float
 * @param exit3_x	message variable of type float
 * @param exit3_y	message variable of type float
 * @param exit4_x	message variable of type float
 * @param exit4_y	message variable of type float
 * @param exit5_x	message variable of type float
 * @param exit5_y	message variable of type float
 * @param exit6_x	message variable of type float
 * @param exit6_y	message variable of type float
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int x, int y, int exit_no, float height, float collision_x, float collision_y, float exit0_x, float exit0_y, float exit1_x, float exit1_y, float exit2_x, float exit2_y, float exit3_x, float exit3_y, float exit4_x, float exit4_y, float exit5_x, float exit5_y, float exit6_x, float exit6_y);
 
/** get_first_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param navmap_cell_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_first_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int agentx, int agent_y);

/** get_next_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param navmap_cell_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_next_navmap_cell_message(xmachine_message_navmap_cell* current, xmachine_message_navmap_cell_list* navmap_cell_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_agent_agent
 * Adds a new continuous valued agent agent to the xmachine_memory_agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_agent_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param velx	agent agent variable of type float
 * @param vely	agent agent variable of type float
 * @param steer_x	agent agent variable of type float
 * @param steer_y	agent agent variable of type float
 * @param height	agent agent variable of type float
 * @param exit_no	agent agent variable of type int
 * @param speed	agent agent variable of type float
 * @param lod	agent agent variable of type int
 * @param animate	agent agent variable of type float
 * @param animate_dir	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_agent_agent(xmachine_memory_agent_list* agents, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir);


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_agents Pointer to agent list on the host
 * @param d_agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param d_navmaps Pointer to agent list on the GPU device
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_agents Pointer to agent list on the host
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_agent_MAX_count
 * Gets the max agent count for the agent agent type 
 * @return		the maximum agent agent count
 */
extern int get_agent_agent_MAX_count();



/** get_agent_agent_default_count
 * Gets the agent count for the agent agent type in state default
 * @return		the current agent agent count in state default
 */
extern int get_agent_agent_default_count();

/** reset_default_count
 * Resets the agent count of the agent in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_agent_default_count();

/** get_device_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the GPU device
 * @return		a xmachine_memory_agent_list on the GPU device
 */
extern xmachine_memory_agent_list* get_device_agent_default_agents();

/** get_host_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the CPU host
 * @return		a xmachine_memory_agent_list on the CPU host
 */
extern xmachine_memory_agent_list* get_host_agent_default_agents();


/** sort_agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_list* agents));


    
/** get_agent_navmap_MAX_count
 * Gets the max agent count for the navmap agent type 
 * @return		the maximum navmap agent count
 */
extern int get_agent_navmap_MAX_count();



/** get_agent_navmap_static_count
 * Gets the agent count for the navmap agent type in state static
 * @return		the current navmap agent count in state static
 */
extern int get_agent_navmap_static_count();

/** reset_static_count
 * Resets the agent count of the navmap in state static to 0. This is useful for interacting with some visualisations.
 */
extern void reset_navmap_static_count();

/** get_device_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the GPU device
 * @return		a xmachine_memory_navmap_list on the GPU device
 */
extern xmachine_memory_navmap_list* get_device_navmap_static_agents();

/** get_host_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the CPU host
 * @return		a xmachine_memory_navmap_list on the CPU host
 */
extern xmachine_memory_navmap_list* get_host_navmap_static_agents();


/** get_navmap_population_width
 * Gets an int value representing the xmachine_memory_navmap population width.
 * @return		xmachine_memory_navmap population width
 */
extern int get_navmap_population_width();


/* Host based access of agent variables*/

/** float get_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_agent_default_variable_x(unsigned int index);

/** float get_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_agent_default_variable_y(unsigned int index);

/** float get_agent_default_variable_velx(unsigned int index)
 * Gets the value of the velx variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable velx
 */
__host__ float get_agent_default_variable_velx(unsigned int index);

/** float get_agent_default_variable_vely(unsigned int index)
 * Gets the value of the vely variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable vely
 */
__host__ float get_agent_default_variable_vely(unsigned int index);

/** float get_agent_default_variable_steer_x(unsigned int index)
 * Gets the value of the steer_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_x
 */
__host__ float get_agent_default_variable_steer_x(unsigned int index);

/** float get_agent_default_variable_steer_y(unsigned int index)
 * Gets the value of the steer_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_y
 */
__host__ float get_agent_default_variable_steer_y(unsigned int index);

/** float get_agent_default_variable_height(unsigned int index)
 * Gets the value of the height variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_agent_default_variable_height(unsigned int index);

/** int get_agent_default_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_agent_default_variable_exit_no(unsigned int index);

/** float get_agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_agent_default_variable_speed(unsigned int index);

/** int get_agent_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_agent_default_variable_lod(unsigned int index);

/** float get_agent_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_agent_default_variable_animate(unsigned int index);

/** int get_agent_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_agent_default_variable_animate_dir(unsigned int index);

/** int get_navmap_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_navmap_static_variable_x(unsigned int index);

/** int get_navmap_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_navmap_static_variable_y(unsigned int index);

/** int get_navmap_static_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_navmap_static_variable_exit_no(unsigned int index);

/** float get_navmap_static_variable_height(unsigned int index)
 * Gets the value of the height variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_navmap_static_variable_height(unsigned int index);

/** float get_navmap_static_variable_collision_x(unsigned int index)
 * Gets the value of the collision_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_x
 */
__host__ float get_navmap_static_variable_collision_x(unsigned int index);

/** float get_navmap_static_variable_collision_y(unsigned int index)
 * Gets the value of the collision_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_y
 */
__host__ float get_navmap_static_variable_collision_y(unsigned int index);

/** float get_navmap_static_variable_exit0_x(unsigned int index)
 * Gets the value of the exit0_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_x
 */
__host__ float get_navmap_static_variable_exit0_x(unsigned int index);

/** float get_navmap_static_variable_exit0_y(unsigned int index)
 * Gets the value of the exit0_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_y
 */
__host__ float get_navmap_static_variable_exit0_y(unsigned int index);

/** float get_navmap_static_variable_exit1_x(unsigned int index)
 * Gets the value of the exit1_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_x
 */
__host__ float get_navmap_static_variable_exit1_x(unsigned int index);

/** float get_navmap_static_variable_exit1_y(unsigned int index)
 * Gets the value of the exit1_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_y
 */
__host__ float get_navmap_static_variable_exit1_y(unsigned int index);

/** float get_navmap_static_variable_exit2_x(unsigned int index)
 * Gets the value of the exit2_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_x
 */
__host__ float get_navmap_static_variable_exit2_x(unsigned int index);

/** float get_navmap_static_variable_exit2_y(unsigned int index)
 * Gets the value of the exit2_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_y
 */
__host__ float get_navmap_static_variable_exit2_y(unsigned int index);

/** float get_navmap_static_variable_exit3_x(unsigned int index)
 * Gets the value of the exit3_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_x
 */
__host__ float get_navmap_static_variable_exit3_x(unsigned int index);

/** float get_navmap_static_variable_exit3_y(unsigned int index)
 * Gets the value of the exit3_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_y
 */
__host__ float get_navmap_static_variable_exit3_y(unsigned int index);

/** float get_navmap_static_variable_exit4_x(unsigned int index)
 * Gets the value of the exit4_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_x
 */
__host__ float get_navmap_static_variable_exit4_x(unsigned int index);

/** float get_navmap_static_variable_exit4_y(unsigned int index)
 * Gets the value of the exit4_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_y
 */
__host__ float get_navmap_static_variable_exit4_y(unsigned int index);

/** float get_navmap_static_variable_exit5_x(unsigned int index)
 * Gets the value of the exit5_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_x
 */
__host__ float get_navmap_static_variable_exit5_x(unsigned int index);

/** float get_navmap_static_variable_exit5_y(unsigned int index)
 * Gets the value of the exit5_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_y
 */
__host__ float get_navmap_static_variable_exit5_y(unsigned int index);

/** float get_navmap_static_variable_exit6_x(unsigned int index)
 * Gets the value of the exit6_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_x
 */
__host__ float get_navmap_static_variable_exit6_x(unsigned int index);

/** float get_navmap_static_variable_exit6_y(unsigned int index)
 * Gets the value of the exit6_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_y
 */
__host__ float get_navmap_static_variable_exit6_y(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_agent
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated agent struct.
 */
xmachine_memory_agent* h_allocate_agent_agent();
/** h_free_agent_agent
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_agent(xmachine_memory_agent** agent);
/** h_allocate_agent_agent_array
 * Utility function to allocate an array of structs for  agent agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_agent** h_allocate_agent_agent_array(unsigned int count);
/** h_free_agent_agent_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_agent_array(xmachine_memory_agent*** agents, unsigned int count);


/** h_add_agent_agent_default
 * Host function to add a single agent of type agent to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_agent_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_agent_default(xmachine_memory_agent* agent);

/** h_add_agents_agent_default(
 * Host function to add multiple agents of type agent to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_agent_default(xmachine_memory_agent** agents, unsigned int count);

/** h_allocate_agent_navmap
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated navmap struct.
 */
xmachine_memory_navmap* h_allocate_agent_navmap();
/** h_free_agent_navmap
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_navmap(xmachine_memory_navmap** agent);
/** h_allocate_agent_navmap_array
 * Utility function to allocate an array of structs for  navmap agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_navmap** h_allocate_agent_navmap_array(unsigned int count);
/** h_free_agent_navmap_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_navmap_array(xmachine_memory_navmap*** agents, unsigned int count);


/** h_add_agent_navmap_static
 * Host function to add a single agent of type navmap to the static state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_navmap_static instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_navmap_static(xmachine_memory_navmap* agent);

/** h_add_agents_navmap_static(
 * Host function to add multiple agents of type navmap to the static state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of navmap agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_navmap_static(xmachine_memory_navmap** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** float reduce_agent_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_x_variable();



/** float min_agent_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_x_variable();
/** float max_agent_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_x_variable();

/** float reduce_agent_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_y_variable();



/** float min_agent_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_y_variable();
/** float max_agent_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_y_variable();

/** float reduce_agent_default_velx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_velx_variable();



/** float min_agent_default_velx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_velx_variable();
/** float max_agent_default_velx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_velx_variable();

/** float reduce_agent_default_vely_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_vely_variable();



/** float min_agent_default_vely_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_vely_variable();
/** float max_agent_default_vely_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_vely_variable();

/** float reduce_agent_default_steer_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_steer_x_variable();



/** float min_agent_default_steer_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_steer_x_variable();
/** float max_agent_default_steer_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_steer_x_variable();

/** float reduce_agent_default_steer_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_steer_y_variable();



/** float min_agent_default_steer_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_steer_y_variable();
/** float max_agent_default_steer_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_steer_y_variable();

/** float reduce_agent_default_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_height_variable();



/** float min_agent_default_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_height_variable();
/** float max_agent_default_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_height_variable();

/** int reduce_agent_default_exit_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_exit_no_variable();



/** int count_agent_default_exit_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_exit_no_variable(int count_value);

/** int min_agent_default_exit_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_exit_no_variable();
/** int max_agent_default_exit_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_exit_no_variable();

/** float reduce_agent_default_speed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_speed_variable();



/** float min_agent_default_speed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_speed_variable();
/** float max_agent_default_speed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_speed_variable();

/** int reduce_agent_default_lod_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_lod_variable();



/** int count_agent_default_lod_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_lod_variable(int count_value);

/** int min_agent_default_lod_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_lod_variable();
/** int max_agent_default_lod_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_lod_variable();

/** float reduce_agent_default_animate_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_animate_variable();



/** float min_agent_default_animate_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_animate_variable();
/** float max_agent_default_animate_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_animate_variable();

/** int reduce_agent_default_animate_dir_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_animate_dir_variable();



/** int count_agent_default_animate_dir_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_animate_dir_variable(int count_value);

/** int min_agent_default_animate_dir_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_animate_dir_variable();
/** int max_agent_default_animate_dir_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_animate_dir_variable();

/** int reduce_navmap_static_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_x_variable();



/** int count_navmap_static_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_x_variable(int count_value);

/** int min_navmap_static_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_x_variable();
/** int max_navmap_static_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_x_variable();

/** int reduce_navmap_static_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_y_variable();



/** int count_navmap_static_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_y_variable(int count_value);

/** int min_navmap_static_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_y_variable();
/** int max_navmap_static_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_y_variable();

/** int reduce_navmap_static_exit_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_exit_no_variable();



/** int count_navmap_static_exit_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_exit_no_variable(int count_value);

/** int min_navmap_static_exit_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_exit_no_variable();
/** int max_navmap_static_exit_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_exit_no_variable();

/** float reduce_navmap_static_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_height_variable();



/** float min_navmap_static_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_height_variable();
/** float max_navmap_static_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_height_variable();

/** float reduce_navmap_static_collision_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_collision_x_variable();



/** float min_navmap_static_collision_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_collision_x_variable();
/** float max_navmap_static_collision_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_collision_x_variable();

/** float reduce_navmap_static_collision_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_collision_y_variable();



/** float min_navmap_static_collision_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_collision_y_variable();
/** float max_navmap_static_collision_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_collision_y_variable();

/** float reduce_navmap_static_exit0_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit0_x_variable();



/** float min_navmap_static_exit0_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit0_x_variable();
/** float max_navmap_static_exit0_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit0_x_variable();

/** float reduce_navmap_static_exit0_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit0_y_variable();



/** float min_navmap_static_exit0_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit0_y_variable();
/** float max_navmap_static_exit0_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit0_y_variable();

/** float reduce_navmap_static_exit1_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit1_x_variable();



/** float min_navmap_static_exit1_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit1_x_variable();
/** float max_navmap_static_exit1_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit1_x_variable();

/** float reduce_navmap_static_exit1_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit1_y_variable();



/** float min_navmap_static_exit1_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit1_y_variable();
/** float max_navmap_static_exit1_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit1_y_variable();

/** float reduce_navmap_static_exit2_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit2_x_variable();



/** float min_navmap_static_exit2_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit2_x_variable();
/** float max_navmap_static_exit2_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit2_x_variable();

/** float reduce_navmap_static_exit2_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit2_y_variable();



/** float min_navmap_static_exit2_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit2_y_variable();
/** float max_navmap_static_exit2_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit2_y_variable();

/** float reduce_navmap_static_exit3_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit3_x_variable();



/** float min_navmap_static_exit3_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit3_x_variable();
/** float max_navmap_static_exit3_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit3_x_variable();

/** float reduce_navmap_static_exit3_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit3_y_variable();



/** float min_navmap_static_exit3_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit3_y_variable();
/** float max_navmap_static_exit3_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit3_y_variable();

/** float reduce_navmap_static_exit4_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit4_x_variable();



/** float min_navmap_static_exit4_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit4_x_variable();
/** float max_navmap_static_exit4_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit4_x_variable();

/** float reduce_navmap_static_exit4_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit4_y_variable();



/** float min_navmap_static_exit4_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit4_y_variable();
/** float max_navmap_static_exit4_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit4_y_variable();

/** float reduce_navmap_static_exit5_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit5_x_variable();



/** float min_navmap_static_exit5_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit5_x_variable();
/** float max_navmap_static_exit5_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit5_x_variable();

/** float reduce_navmap_static_exit5_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit5_y_variable();



/** float min_navmap_static_exit5_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit5_y_variable();
/** float max_navmap_static_exit5_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit5_y_variable();

/** float reduce_navmap_static_exit6_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit6_x_variable();



/** float min_navmap_static_exit6_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit6_x_variable();
/** float max_navmap_static_exit6_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit6_x_variable();

/** float reduce_navmap_static_exit6_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit6_y_variable();



/** float min_navmap_static_exit6_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit6_y_variable();
/** float max_navmap_static_exit6_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit6_y_variable();


  
/* global constant variables */

__constant__ float EMMISION_RATE_EXIT1;

__constant__ float EMMISION_RATE_EXIT2;

__constant__ float EMMISION_RATE_EXIT3;

__constant__ float EMMISION_RATE_EXIT4;

__constant__ float EMMISION_RATE_EXIT5;

__constant__ float EMMISION_RATE_EXIT6;

__constant__ float EMMISION_RATE_EXIT7;

__constant__ int EXIT1_PROBABILITY;

__constant__ int EXIT2_PROBABILITY;

__constant__ int EXIT3_PROBABILITY;

__constant__ int EXIT4_PROBABILITY;

__constant__ int EXIT5_PROBABILITY;

__constant__ int EXIT6_PROBABILITY;

__constant__ int EXIT7_PROBABILITY;

__constant__ int EXIT1_STATE;

__constant__ int EXIT2_STATE;

__constant__ int EXIT3_STATE;

__constant__ int EXIT4_STATE;

__constant__ int EXIT5_STATE;

__constant__ int EXIT6_STATE;

__constant__ int EXIT7_STATE;

__constant__ int EXIT1_CELL_COUNT;

__constant__ int EXIT2_CELL_COUNT;

__constant__ int EXIT3_CELL_COUNT;

__constant__ int EXIT4_CELL_COUNT;

__constant__ int EXIT5_CELL_COUNT;

__constant__ int EXIT6_CELL_COUNT;

__constant__ int EXIT7_CELL_COUNT;

__constant__ float TIME_SCALER;

__constant__ float STEER_WEIGHT;

__constant__ float AVOID_WEIGHT;

__constant__ float COLLISION_WEIGHT;

__constant__ float GOAL_WEIGHT;

/** set_EMMISION_RATE_EXIT1
 * Sets the constant variable EMMISION_RATE_EXIT1 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT1 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE_EXIT1);

extern const float* get_EMMISION_RATE_EXIT1();


extern float h_env_EMMISION_RATE_EXIT1;

/** set_EMMISION_RATE_EXIT2
 * Sets the constant variable EMMISION_RATE_EXIT2 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT2 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE_EXIT2);

extern const float* get_EMMISION_RATE_EXIT2();


extern float h_env_EMMISION_RATE_EXIT2;

/** set_EMMISION_RATE_EXIT3
 * Sets the constant variable EMMISION_RATE_EXIT3 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT3 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE_EXIT3);

extern const float* get_EMMISION_RATE_EXIT3();


extern float h_env_EMMISION_RATE_EXIT3;

/** set_EMMISION_RATE_EXIT4
 * Sets the constant variable EMMISION_RATE_EXIT4 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT4 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE_EXIT4);

extern const float* get_EMMISION_RATE_EXIT4();


extern float h_env_EMMISION_RATE_EXIT4;

/** set_EMMISION_RATE_EXIT5
 * Sets the constant variable EMMISION_RATE_EXIT5 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT5 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE_EXIT5);

extern const float* get_EMMISION_RATE_EXIT5();


extern float h_env_EMMISION_RATE_EXIT5;

/** set_EMMISION_RATE_EXIT6
 * Sets the constant variable EMMISION_RATE_EXIT6 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT6 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE_EXIT6);

extern const float* get_EMMISION_RATE_EXIT6();


extern float h_env_EMMISION_RATE_EXIT6;

/** set_EMMISION_RATE_EXIT7
 * Sets the constant variable EMMISION_RATE_EXIT7 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT7 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE_EXIT7);

extern const float* get_EMMISION_RATE_EXIT7();


extern float h_env_EMMISION_RATE_EXIT7;

/** set_EXIT1_PROBABILITY
 * Sets the constant variable EXIT1_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT1_PROBABILITY value to set the variable
 */
extern void set_EXIT1_PROBABILITY(int* h_EXIT1_PROBABILITY);

extern const int* get_EXIT1_PROBABILITY();


extern int h_env_EXIT1_PROBABILITY;

/** set_EXIT2_PROBABILITY
 * Sets the constant variable EXIT2_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT2_PROBABILITY value to set the variable
 */
extern void set_EXIT2_PROBABILITY(int* h_EXIT2_PROBABILITY);

extern const int* get_EXIT2_PROBABILITY();


extern int h_env_EXIT2_PROBABILITY;

/** set_EXIT3_PROBABILITY
 * Sets the constant variable EXIT3_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT3_PROBABILITY value to set the variable
 */
extern void set_EXIT3_PROBABILITY(int* h_EXIT3_PROBABILITY);

extern const int* get_EXIT3_PROBABILITY();


extern int h_env_EXIT3_PROBABILITY;

/** set_EXIT4_PROBABILITY
 * Sets the constant variable EXIT4_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT4_PROBABILITY value to set the variable
 */
extern void set_EXIT4_PROBABILITY(int* h_EXIT4_PROBABILITY);

extern const int* get_EXIT4_PROBABILITY();


extern int h_env_EXIT4_PROBABILITY;

/** set_EXIT5_PROBABILITY
 * Sets the constant variable EXIT5_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT5_PROBABILITY value to set the variable
 */
extern void set_EXIT5_PROBABILITY(int* h_EXIT5_PROBABILITY);

extern const int* get_EXIT5_PROBABILITY();


extern int h_env_EXIT5_PROBABILITY;

/** set_EXIT6_PROBABILITY
 * Sets the constant variable EXIT6_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT6_PROBABILITY value to set the variable
 */
extern void set_EXIT6_PROBABILITY(int* h_EXIT6_PROBABILITY);

extern const int* get_EXIT6_PROBABILITY();


extern int h_env_EXIT6_PROBABILITY;

/** set_EXIT7_PROBABILITY
 * Sets the constant variable EXIT7_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT7_PROBABILITY value to set the variable
 */
extern void set_EXIT7_PROBABILITY(int* h_EXIT7_PROBABILITY);

extern const int* get_EXIT7_PROBABILITY();


extern int h_env_EXIT7_PROBABILITY;

/** set_EXIT1_STATE
 * Sets the constant variable EXIT1_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT1_STATE value to set the variable
 */
extern void set_EXIT1_STATE(int* h_EXIT1_STATE);

extern const int* get_EXIT1_STATE();


extern int h_env_EXIT1_STATE;

/** set_EXIT2_STATE
 * Sets the constant variable EXIT2_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT2_STATE value to set the variable
 */
extern void set_EXIT2_STATE(int* h_EXIT2_STATE);

extern const int* get_EXIT2_STATE();


extern int h_env_EXIT2_STATE;

/** set_EXIT3_STATE
 * Sets the constant variable EXIT3_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT3_STATE value to set the variable
 */
extern void set_EXIT3_STATE(int* h_EXIT3_STATE);

extern const int* get_EXIT3_STATE();


extern int h_env_EXIT3_STATE;

/** set_EXIT4_STATE
 * Sets the constant variable EXIT4_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT4_STATE value to set the variable
 */
extern void set_EXIT4_STATE(int* h_EXIT4_STATE);

extern const int* get_EXIT4_STATE();


extern int h_env_EXIT4_STATE;

/** set_EXIT5_STATE
 * Sets the constant variable EXIT5_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT5_STATE value to set the variable
 */
extern void set_EXIT5_STATE(int* h_EXIT5_STATE);

extern const int* get_EXIT5_STATE();


extern int h_env_EXIT5_STATE;

/** set_EXIT6_STATE
 * Sets the constant variable EXIT6_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT6_STATE value to set the variable
 */
extern void set_EXIT6_STATE(int* h_EXIT6_STATE);

extern const int* get_EXIT6_STATE();


extern int h_env_EXIT6_STATE;

/** set_EXIT7_STATE
 * Sets the constant variable EXIT7_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT7_STATE value to set the variable
 */
extern void set_EXIT7_STATE(int* h_EXIT7_STATE);

extern const int* get_EXIT7_STATE();


extern int h_env_EXIT7_STATE;

/** set_EXIT1_CELL_COUNT
 * Sets the constant variable EXIT1_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT1_CELL_COUNT value to set the variable
 */
extern void set_EXIT1_CELL_COUNT(int* h_EXIT1_CELL_COUNT);

extern const int* get_EXIT1_CELL_COUNT();


extern int h_env_EXIT1_CELL_COUNT;

/** set_EXIT2_CELL_COUNT
 * Sets the constant variable EXIT2_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT2_CELL_COUNT value to set the variable
 */
extern void set_EXIT2_CELL_COUNT(int* h_EXIT2_CELL_COUNT);

extern const int* get_EXIT2_CELL_COUNT();


extern int h_env_EXIT2_CELL_COUNT;

/** set_EXIT3_CELL_COUNT
 * Sets the constant variable EXIT3_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT3_CELL_COUNT value to set the variable
 */
extern void set_EXIT3_CELL_COUNT(int* h_EXIT3_CELL_COUNT);

extern const int* get_EXIT3_CELL_COUNT();


extern int h_env_EXIT3_CELL_COUNT;

/** set_EXIT4_CELL_COUNT
 * Sets the constant variable EXIT4_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT4_CELL_COUNT value to set the variable
 */
extern void set_EXIT4_CELL_COUNT(int* h_EXIT4_CELL_COUNT);

extern const int* get_EXIT4_CELL_COUNT();


extern int h_env_EXIT4_CELL_COUNT;

/** set_EXIT5_CELL_COUNT
 * Sets the constant variable EXIT5_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT5_CELL_COUNT value to set the variable
 */
extern void set_EXIT5_CELL_COUNT(int* h_EXIT5_CELL_COUNT);

extern const int* get_EXIT5_CELL_COUNT();


extern int h_env_EXIT5_CELL_COUNT;

/** set_EXIT6_CELL_COUNT
 * Sets the constant variable EXIT6_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT6_CELL_COUNT value to set the variable
 */
extern void set_EXIT6_CELL_COUNT(int* h_EXIT6_CELL_COUNT);

extern const int* get_EXIT6_CELL_COUNT();


extern int h_env_EXIT6_CELL_COUNT;

/** set_EXIT7_CELL_COUNT
 * Sets the constant variable EXIT7_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT7_CELL_COUNT value to set the variable
 */
extern void set_EXIT7_CELL_COUNT(int* h_EXIT7_CELL_COUNT);

extern const int* get_EXIT7_CELL_COUNT();


extern int h_env_EXIT7_CELL_COUNT;

/** set_TIME_SCALER
 * Sets the constant variable TIME_SCALER on the device which can then be used in the agent functions.
 * @param h_TIME_SCALER value to set the variable
 */
extern void set_TIME_SCALER(float* h_TIME_SCALER);

extern const float* get_TIME_SCALER();


extern float h_env_TIME_SCALER;

/** set_STEER_WEIGHT
 * Sets the constant variable STEER_WEIGHT on the device which can then be used in the agent functions.
 * @param h_STEER_WEIGHT value to set the variable
 */
extern void set_STEER_WEIGHT(float* h_STEER_WEIGHT);

extern const float* get_STEER_WEIGHT();


extern float h_env_STEER_WEIGHT;

/** set_AVOID_WEIGHT
 * Sets the constant variable AVOID_WEIGHT on the device which can then be used in the agent functions.
 * @param h_AVOID_WEIGHT value to set the variable
 */
extern void set_AVOID_WEIGHT(float* h_AVOID_WEIGHT);

extern const float* get_AVOID_WEIGHT();


extern float h_env_AVOID_WEIGHT;

/** set_COLLISION_WEIGHT
 * Sets the constant variable COLLISION_WEIGHT on the device which can then be used in the agent functions.
 * @param h_COLLISION_WEIGHT value to set the variable
 */
extern void set_COLLISION_WEIGHT(float* h_COLLISION_WEIGHT);

extern const float* get_COLLISION_WEIGHT();


extern float h_env_COLLISION_WEIGHT;

/** set_GOAL_WEIGHT
 * Sets the constant variable GOAL_WEIGHT on the device which can then be used in the agent functions.
 * @param h_GOAL_WEIGHT value to set the variable
 */
extern void set_GOAL_WEIGHT(float* h_GOAL_WEIGHT);

extern const float* get_GOAL_WEIGHT();


extern float h_env_GOAL_WEIGHT;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

