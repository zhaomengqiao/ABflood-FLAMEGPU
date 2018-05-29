
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

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

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

	
//if this is defined then the project must be built with sm_13 or later
#define _DOUBLE_SUPPORT_REQUIRED_

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 16384

//Maximum population size of xmachine_memory_FloodCell
#define xmachine_memory_FloodCell_MAX 16384


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_WetDryMessage
#define xmachine_message_WetDryMessage_MAX 16384

//Maximum population size of xmachine_mmessage_SpaceOperatorMessage
#define xmachine_message_SpaceOperatorMessage_MAX 16384


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_WetDryMessage_partitioningDiscrete
#define xmachine_message_SpaceOperatorMessage_partitioningDiscrete

/* Spatial partitioning grid size definitions */
  

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

/** struct xmachine_memory_FloodCell
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_FloodCell
{
    int inDomain;    /**< X-machine memory variable inDomain of type int.*/
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    double z0;    /**< X-machine memory variable z0 of type double.*/
    double h;    /**< X-machine memory variable h of type double.*/
    double qx;    /**< X-machine memory variable qx of type double.*/
    double qy;    /**< X-machine memory variable qy of type double.*/
    double timeStep;    /**< X-machine memory variable timeStep of type double.*/
    double minh_loc;    /**< X-machine memory variable minh_loc of type double.*/
    double hFace_E;    /**< X-machine memory variable hFace_E of type double.*/
    double etFace_E;    /**< X-machine memory variable etFace_E of type double.*/
    double qxFace_E;    /**< X-machine memory variable qxFace_E of type double.*/
    double qyFace_E;    /**< X-machine memory variable qyFace_E of type double.*/
    double hFace_W;    /**< X-machine memory variable hFace_W of type double.*/
    double etFace_W;    /**< X-machine memory variable etFace_W of type double.*/
    double qxFace_W;    /**< X-machine memory variable qxFace_W of type double.*/
    double qyFace_W;    /**< X-machine memory variable qyFace_W of type double.*/
    double hFace_N;    /**< X-machine memory variable hFace_N of type double.*/
    double etFace_N;    /**< X-machine memory variable etFace_N of type double.*/
    double qxFace_N;    /**< X-machine memory variable qxFace_N of type double.*/
    double qyFace_N;    /**< X-machine memory variable qyFace_N of type double.*/
    double hFace_S;    /**< X-machine memory variable hFace_S of type double.*/
    double etFace_S;    /**< X-machine memory variable etFace_S of type double.*/
    double qxFace_S;    /**< X-machine memory variable qxFace_S of type double.*/
    double qyFace_S;    /**< X-machine memory variable qyFace_S of type double.*/
};



/* Message structures */

/** struct xmachine_message_WetDryMessage
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_WetDryMessage
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int inDomain;        /**< Message variable inDomain of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double min_hloc;        /**< Message variable min_hloc of type double.*/
};

/** struct xmachine_message_SpaceOperatorMessage
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_SpaceOperatorMessage
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int inDomain;        /**< Message variable inDomain of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    double hFace_E;        /**< Message variable hFace_E of type double.*/  
    double etFace_E;        /**< Message variable etFace_E of type double.*/  
    double qFace_X_E;        /**< Message variable qFace_X_E of type double.*/  
    double qFace_Y_E;        /**< Message variable qFace_Y_E of type double.*/  
    double hFace_W;        /**< Message variable hFace_W of type double.*/  
    double etFace_W;        /**< Message variable etFace_W of type double.*/  
    double qFace_X_W;        /**< Message variable qFace_X_W of type double.*/  
    double qFace_Y_W;        /**< Message variable qFace_Y_W of type double.*/  
    double hFace_N;        /**< Message variable hFace_N of type double.*/  
    double etFace_N;        /**< Message variable etFace_N of type double.*/  
    double qFace_X_N;        /**< Message variable qFace_X_N of type double.*/  
    double qFace_Y_N;        /**< Message variable qFace_Y_N of type double.*/  
    double hFace_S;        /**< Message variable hFace_S of type double.*/  
    double etFace_S;        /**< Message variable etFace_S of type double.*/  
    double qFace_X_S;        /**< Message variable qFace_X_S of type double.*/  
    double qFace_Y_S;        /**< Message variable qFace_Y_S of type double.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_FloodCell_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_FloodCell_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_FloodCell_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_FloodCell_MAX];  /**< Used during parallel prefix sum */
    
    int inDomain [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list inDomain of type int.*/
    int x [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list y of type int.*/
    double z0 [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list z0 of type double.*/
    double h [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list h of type double.*/
    double qx [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qx of type double.*/
    double qy [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qy of type double.*/
    double timeStep [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list timeStep of type double.*/
    double minh_loc [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list minh_loc of type double.*/
    double hFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_E of type double.*/
    double etFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_E of type double.*/
    double qxFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_E of type double.*/
    double qyFace_E [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_E of type double.*/
    double hFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_W of type double.*/
    double etFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_W of type double.*/
    double qxFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_W of type double.*/
    double qyFace_W [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_W of type double.*/
    double hFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_N of type double.*/
    double etFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_N of type double.*/
    double qxFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_N of type double.*/
    double qyFace_N [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_N of type double.*/
    double hFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list hFace_S of type double.*/
    double etFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list etFace_S of type double.*/
    double qxFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qxFace_S of type double.*/
    double qyFace_S [xmachine_memory_FloodCell_MAX];    /**< X-machine memory variable list qyFace_S of type double.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_WetDryMessage_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_WetDryMessage_list
{
    int inDomain [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list inDomain of type int.*/
    int x [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list y of type int.*/
    double min_hloc [xmachine_message_WetDryMessage_MAX];    /**< Message memory variable list min_hloc of type double.*/
    
};

/** struct xmachine_message_SpaceOperatorMessage_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_SpaceOperatorMessage_list
{
    int inDomain [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list inDomain of type int.*/
    int x [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list y of type int.*/
    double hFace_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_E of type double.*/
    double etFace_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_E of type double.*/
    double qFace_X_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_E of type double.*/
    double qFace_Y_E [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_E of type double.*/
    double hFace_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_W of type double.*/
    double etFace_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_W of type double.*/
    double qFace_X_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_W of type double.*/
    double qFace_Y_W [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_W of type double.*/
    double hFace_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_N of type double.*/
    double etFace_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_N of type double.*/
    double qFace_X_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_N of type double.*/
    double qFace_Y_N [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_N of type double.*/
    double hFace_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list hFace_S of type double.*/
    double etFace_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list etFace_S of type double.*/
    double qFace_X_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_X_S of type double.*/
    double qFace_Y_S [xmachine_message_SpaceOperatorMessage_MAX];    /**< Message memory variable list qFace_Y_S of type double.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



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
 * PrepareWetDry FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param WetDryMessage_messages Pointer to output message list of type xmachine_message_WetDryMessage_list. Must be passed as an argument to the add_WetDryMessage_message function ??.
 */
__FLAME_GPU_FUNC__ int PrepareWetDry(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages);

/**
 * ProcessWetDryMessage FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param WetDryMessage_messages  WetDryMessage_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_WetDryMessage_message and get_next_WetDryMessage_message functions.
 */
__FLAME_GPU_FUNC__ int ProcessWetDryMessage(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages);

/**
 * PrepareSpaceOperator FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param SpaceOperatorMessage_messages Pointer to output message list of type xmachine_message_SpaceOperatorMessage_list. Must be passed as an argument to the add_SpaceOperatorMessage_message function ??.
 */
__FLAME_GPU_FUNC__ int PrepareSpaceOperator(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages);

/**
 * ProcessSpaceOperatorMessage FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_FloodCell. This represents a single agent instance and can be modified directly.
 * @param SpaceOperatorMessage_messages  SpaceOperatorMessage_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_SpaceOperatorMessage_message and get_next_SpaceOperatorMessage_message functions.
 */
__FLAME_GPU_FUNC__ int ProcessSpaceOperatorMessage(xmachine_memory_FloodCell* agent, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages);

  
/* Message Function Prototypes for Discrete Partitioned WetDryMessage message implemented in FLAMEGPU_Kernels */

/** add_WetDryMessage_message
 * Function for all types of message partitioning
 * Adds a new WetDryMessage agent to the xmachine_memory_WetDryMessage_list list using a linear mapping
 * @param agents	xmachine_memory_WetDryMessage_list agent list
 * @param inDomain	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param min_hloc	message variable of type double
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_WetDryMessage_message(xmachine_message_WetDryMessage_list* WetDryMessage_messages, int inDomain, int x, int y, double min_hloc);
 
/** get_first_WetDryMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param WetDryMessage_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_WetDryMessage * get_first_WetDryMessage_message(xmachine_message_WetDryMessage_list* WetDryMessage_messages, int agentx, int agent_y);

/** get_next_WetDryMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param WetDryMessage_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_WetDryMessage * get_next_WetDryMessage_message(xmachine_message_WetDryMessage* current, xmachine_message_WetDryMessage_list* WetDryMessage_messages);

  
/* Message Function Prototypes for Discrete Partitioned SpaceOperatorMessage message implemented in FLAMEGPU_Kernels */

/** add_SpaceOperatorMessage_message
 * Function for all types of message partitioning
 * Adds a new SpaceOperatorMessage agent to the xmachine_memory_SpaceOperatorMessage_list list using a linear mapping
 * @param agents	xmachine_memory_SpaceOperatorMessage_list agent list
 * @param inDomain	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param hFace_E	message variable of type double
 * @param etFace_E	message variable of type double
 * @param qFace_X_E	message variable of type double
 * @param qFace_Y_E	message variable of type double
 * @param hFace_W	message variable of type double
 * @param etFace_W	message variable of type double
 * @param qFace_X_W	message variable of type double
 * @param qFace_Y_W	message variable of type double
 * @param hFace_N	message variable of type double
 * @param etFace_N	message variable of type double
 * @param qFace_X_N	message variable of type double
 * @param qFace_Y_N	message variable of type double
 * @param hFace_S	message variable of type double
 * @param etFace_S	message variable of type double
 * @param qFace_X_S	message variable of type double
 * @param qFace_Y_S	message variable of type double
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages, int inDomain, int x, int y, double hFace_E, double etFace_E, double qFace_X_E, double qFace_Y_E, double hFace_W, double etFace_W, double qFace_X_W, double qFace_Y_W, double hFace_N, double etFace_N, double qFace_X_N, double qFace_Y_N, double hFace_S, double etFace_S, double qFace_X_S, double qFace_Y_S);
 
/** get_first_SpaceOperatorMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param SpaceOperatorMessage_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_SpaceOperatorMessage * get_first_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages, int agentx, int agent_y);

/** get_next_SpaceOperatorMessage_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param SpaceOperatorMessage_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_SpaceOperatorMessage * get_next_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage* current, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */


  
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
 * @param h_FloodCells Pointer to agent list on the host
 * @param d_FloodCells Pointer to agent list on the GPU device
 * @param h_xmachine_memory_FloodCell_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_FloodCell_list* h_FloodCells_Default, xmachine_memory_FloodCell_list* d_FloodCells_Default, int h_xmachine_memory_FloodCell_Default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_FloodCells Pointer to agent list on the host
 * @param h_xmachine_memory_FloodCell_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_FloodCell_list* h_FloodCells, int* h_xmachine_memory_FloodCell_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_FloodCell_MAX_count
 * Gets the max agent count for the FloodCell agent type 
 * @return		the maximum FloodCell agent count
 */
extern int get_agent_FloodCell_MAX_count();



/** get_agent_FloodCell_Default_count
 * Gets the agent count for the FloodCell agent type in state Default
 * @return		the current FloodCell agent count in state Default
 */
extern int get_agent_FloodCell_Default_count();

/** reset_Default_count
 * Resets the agent count of the FloodCell in state Default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_FloodCell_Default_count();

/** get_device_FloodCell_Default_agents
 * Gets a pointer to xmachine_memory_FloodCell_list on the GPU device
 * @return		a xmachine_memory_FloodCell_list on the GPU device
 */
extern xmachine_memory_FloodCell_list* get_device_FloodCell_Default_agents();

/** get_host_FloodCell_Default_agents
 * Gets a pointer to xmachine_memory_FloodCell_list on the CPU host
 * @return		a xmachine_memory_FloodCell_list on the CPU host
 */
extern xmachine_memory_FloodCell_list* get_host_FloodCell_Default_agents();


/** get_FloodCell_population_width
 * Gets an int value representing the xmachine_memory_FloodCell population width.
 * @return		xmachine_memory_FloodCell population width
 */
extern int get_FloodCell_population_width();


/* Host based access of agent variables*/

/** int get_FloodCell_Default_variable_inDomain(unsigned int index)
 * Gets the value of the inDomain variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable inDomain
 */
__host__ int get_FloodCell_Default_variable_inDomain(unsigned int index);

/** int get_FloodCell_Default_variable_x(unsigned int index)
 * Gets the value of the x variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_FloodCell_Default_variable_x(unsigned int index);

/** int get_FloodCell_Default_variable_y(unsigned int index)
 * Gets the value of the y variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_FloodCell_Default_variable_y(unsigned int index);

/** double get_FloodCell_Default_variable_z0(unsigned int index)
 * Gets the value of the z0 variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z0
 */
__host__ double get_FloodCell_Default_variable_z0(unsigned int index);

/** double get_FloodCell_Default_variable_h(unsigned int index)
 * Gets the value of the h variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable h
 */
__host__ double get_FloodCell_Default_variable_h(unsigned int index);

/** double get_FloodCell_Default_variable_qx(unsigned int index)
 * Gets the value of the qx variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qx
 */
__host__ double get_FloodCell_Default_variable_qx(unsigned int index);

/** double get_FloodCell_Default_variable_qy(unsigned int index)
 * Gets the value of the qy variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qy
 */
__host__ double get_FloodCell_Default_variable_qy(unsigned int index);

/** double get_FloodCell_Default_variable_timeStep(unsigned int index)
 * Gets the value of the timeStep variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timeStep
 */
__host__ double get_FloodCell_Default_variable_timeStep(unsigned int index);

/** double get_FloodCell_Default_variable_minh_loc(unsigned int index)
 * Gets the value of the minh_loc variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable minh_loc
 */
__host__ double get_FloodCell_Default_variable_minh_loc(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_E(unsigned int index)
 * Gets the value of the hFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_E
 */
__host__ double get_FloodCell_Default_variable_hFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_E(unsigned int index)
 * Gets the value of the etFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_E
 */
__host__ double get_FloodCell_Default_variable_etFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_E(unsigned int index)
 * Gets the value of the qxFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_E
 */
__host__ double get_FloodCell_Default_variable_qxFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_E(unsigned int index)
 * Gets the value of the qyFace_E variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_E
 */
__host__ double get_FloodCell_Default_variable_qyFace_E(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_W(unsigned int index)
 * Gets the value of the hFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_W
 */
__host__ double get_FloodCell_Default_variable_hFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_W(unsigned int index)
 * Gets the value of the etFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_W
 */
__host__ double get_FloodCell_Default_variable_etFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_W(unsigned int index)
 * Gets the value of the qxFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_W
 */
__host__ double get_FloodCell_Default_variable_qxFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_W(unsigned int index)
 * Gets the value of the qyFace_W variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_W
 */
__host__ double get_FloodCell_Default_variable_qyFace_W(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_N(unsigned int index)
 * Gets the value of the hFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_N
 */
__host__ double get_FloodCell_Default_variable_hFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_N(unsigned int index)
 * Gets the value of the etFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_N
 */
__host__ double get_FloodCell_Default_variable_etFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_N(unsigned int index)
 * Gets the value of the qxFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_N
 */
__host__ double get_FloodCell_Default_variable_qxFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_N(unsigned int index)
 * Gets the value of the qyFace_N variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_N
 */
__host__ double get_FloodCell_Default_variable_qyFace_N(unsigned int index);

/** double get_FloodCell_Default_variable_hFace_S(unsigned int index)
 * Gets the value of the hFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hFace_S
 */
__host__ double get_FloodCell_Default_variable_hFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_etFace_S(unsigned int index)
 * Gets the value of the etFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable etFace_S
 */
__host__ double get_FloodCell_Default_variable_etFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_qxFace_S(unsigned int index)
 * Gets the value of the qxFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qxFace_S
 */
__host__ double get_FloodCell_Default_variable_qxFace_S(unsigned int index);

/** double get_FloodCell_Default_variable_qyFace_S(unsigned int index)
 * Gets the value of the qyFace_S variable of an FloodCell agent in the Default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable qyFace_S
 */
__host__ double get_FloodCell_Default_variable_qyFace_S(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_FloodCell
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated FloodCell struct.
 */
xmachine_memory_FloodCell* h_allocate_agent_FloodCell();
/** h_free_agent_FloodCell
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_FloodCell(xmachine_memory_FloodCell** agent);
/** h_allocate_agent_FloodCell_array
 * Utility function to allocate an array of structs for  FloodCell agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_FloodCell** h_allocate_agent_FloodCell_array(unsigned int count);
/** h_free_agent_FloodCell_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_FloodCell_array(xmachine_memory_FloodCell*** agents, unsigned int count);


/** h_add_agent_FloodCell_Default
 * Host function to add a single agent of type FloodCell to the Default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_FloodCell_Default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_FloodCell_Default(xmachine_memory_FloodCell* agent);

/** h_add_agents_FloodCell_Default(
 * Host function to add multiple agents of type FloodCell to the Default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of FloodCell agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_FloodCell_Default(xmachine_memory_FloodCell** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_FloodCell_Default_inDomain_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_FloodCell_Default_inDomain_variable();



/** int count_FloodCell_Default_inDomain_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_FloodCell_Default_inDomain_variable(int count_value);

/** int min_FloodCell_Default_inDomain_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_FloodCell_Default_inDomain_variable();
/** int max_FloodCell_Default_inDomain_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_FloodCell_Default_inDomain_variable();

/** int reduce_FloodCell_Default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_FloodCell_Default_x_variable();



/** int count_FloodCell_Default_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_FloodCell_Default_x_variable(int count_value);

/** int min_FloodCell_Default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_FloodCell_Default_x_variable();
/** int max_FloodCell_Default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_FloodCell_Default_x_variable();

/** int reduce_FloodCell_Default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_FloodCell_Default_y_variable();



/** int count_FloodCell_Default_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_FloodCell_Default_y_variable(int count_value);

/** int min_FloodCell_Default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_FloodCell_Default_y_variable();
/** int max_FloodCell_Default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_FloodCell_Default_y_variable();

/** double reduce_FloodCell_Default_z0_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_z0_variable();



/** double min_FloodCell_Default_z0_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_z0_variable();
/** double max_FloodCell_Default_z0_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_z0_variable();

/** double reduce_FloodCell_Default_h_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_h_variable();



/** double min_FloodCell_Default_h_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_h_variable();
/** double max_FloodCell_Default_h_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_h_variable();

/** double reduce_FloodCell_Default_qx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qx_variable();



/** double min_FloodCell_Default_qx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qx_variable();
/** double max_FloodCell_Default_qx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qx_variable();

/** double reduce_FloodCell_Default_qy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qy_variable();



/** double min_FloodCell_Default_qy_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qy_variable();
/** double max_FloodCell_Default_qy_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qy_variable();

/** double reduce_FloodCell_Default_timeStep_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_timeStep_variable();



/** double min_FloodCell_Default_timeStep_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_timeStep_variable();
/** double max_FloodCell_Default_timeStep_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_timeStep_variable();

/** double reduce_FloodCell_Default_minh_loc_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_minh_loc_variable();



/** double min_FloodCell_Default_minh_loc_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_minh_loc_variable();
/** double max_FloodCell_Default_minh_loc_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_minh_loc_variable();

/** double reduce_FloodCell_Default_hFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_E_variable();



/** double min_FloodCell_Default_hFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_E_variable();
/** double max_FloodCell_Default_hFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_E_variable();

/** double reduce_FloodCell_Default_etFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_E_variable();



/** double min_FloodCell_Default_etFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_E_variable();
/** double max_FloodCell_Default_etFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_E_variable();

/** double reduce_FloodCell_Default_qxFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_E_variable();



/** double min_FloodCell_Default_qxFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_E_variable();
/** double max_FloodCell_Default_qxFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_E_variable();

/** double reduce_FloodCell_Default_qyFace_E_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_E_variable();



/** double min_FloodCell_Default_qyFace_E_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_E_variable();
/** double max_FloodCell_Default_qyFace_E_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_E_variable();

/** double reduce_FloodCell_Default_hFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_W_variable();



/** double min_FloodCell_Default_hFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_W_variable();
/** double max_FloodCell_Default_hFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_W_variable();

/** double reduce_FloodCell_Default_etFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_W_variable();



/** double min_FloodCell_Default_etFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_W_variable();
/** double max_FloodCell_Default_etFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_W_variable();

/** double reduce_FloodCell_Default_qxFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_W_variable();



/** double min_FloodCell_Default_qxFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_W_variable();
/** double max_FloodCell_Default_qxFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_W_variable();

/** double reduce_FloodCell_Default_qyFace_W_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_W_variable();



/** double min_FloodCell_Default_qyFace_W_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_W_variable();
/** double max_FloodCell_Default_qyFace_W_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_W_variable();

/** double reduce_FloodCell_Default_hFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_N_variable();



/** double min_FloodCell_Default_hFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_N_variable();
/** double max_FloodCell_Default_hFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_N_variable();

/** double reduce_FloodCell_Default_etFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_N_variable();



/** double min_FloodCell_Default_etFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_N_variable();
/** double max_FloodCell_Default_etFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_N_variable();

/** double reduce_FloodCell_Default_qxFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_N_variable();



/** double min_FloodCell_Default_qxFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_N_variable();
/** double max_FloodCell_Default_qxFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_N_variable();

/** double reduce_FloodCell_Default_qyFace_N_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_N_variable();



/** double min_FloodCell_Default_qyFace_N_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_N_variable();
/** double max_FloodCell_Default_qyFace_N_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_N_variable();

/** double reduce_FloodCell_Default_hFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_hFace_S_variable();



/** double min_FloodCell_Default_hFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_hFace_S_variable();
/** double max_FloodCell_Default_hFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_hFace_S_variable();

/** double reduce_FloodCell_Default_etFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_etFace_S_variable();



/** double min_FloodCell_Default_etFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_etFace_S_variable();
/** double max_FloodCell_Default_etFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_etFace_S_variable();

/** double reduce_FloodCell_Default_qxFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qxFace_S_variable();



/** double min_FloodCell_Default_qxFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qxFace_S_variable();
/** double max_FloodCell_Default_qxFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qxFace_S_variable();

/** double reduce_FloodCell_Default_qyFace_S_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
double reduce_FloodCell_Default_qyFace_S_variable();



/** double min_FloodCell_Default_qyFace_S_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double min_FloodCell_Default_qyFace_S_variable();
/** double max_FloodCell_Default_qyFace_S_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
double max_FloodCell_Default_qyFace_S_variable();


  
/* global constant variables */

__constant__ double dt;

/** set_dt
 * Sets the constant variable dt on the device which can then be used in the agent functions.
 * @param h_dt value to set the variable
 */
extern void set_dt(double* h_dt);

extern const double* get_dt();


extern double h_env_dt;


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

