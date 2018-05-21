
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

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 65536

//Maximum population size of xmachine_memory_cell
#define xmachine_memory_cell_MAX 65536


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_state
#define xmachine_message_state_MAX 65536


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_state_partitioningDiscrete

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

/** struct xmachine_memory_cell
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_cell
{
    int state;    /**< X-machine memory variable state of type int.*/
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
};



/* Message structures */

/** struct xmachine_message_state
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_state
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int state;        /**< Message variable state of type int.*/  
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_cell_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_cell_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_cell_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_cell_MAX];  /**< Used during parallel prefix sum */
    
    int state [xmachine_memory_cell_MAX];    /**< X-machine memory variable list state of type int.*/
    int x [xmachine_memory_cell_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_cell_MAX];    /**< X-machine memory variable list y of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_state_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_state_list
{
    int state [xmachine_message_state_MAX];    /**< Message memory variable list state of type int.*/
    int x [xmachine_message_state_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_state_MAX];    /**< Message memory variable list y of type int.*/
    
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
 * output_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_cell. This represents a single agent instance and can be modified directly.
 * @param state_messages Pointer to output message list of type xmachine_message_state_list. Must be passed as an argument to the add_state_message function ??.
 */
__FLAME_GPU_FUNC__ int output_state(xmachine_memory_cell* agent, xmachine_message_state_list* state_messages);

/**
 * update_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_cell. This represents a single agent instance and can be modified directly.
 * @param state_messages  state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_state_message and get_next_state_message functions.
 */
__FLAME_GPU_FUNC__ int update_state(xmachine_memory_cell* agent, xmachine_message_state_list* state_messages);

  
/* Message Function Prototypes for Discrete Partitioned state message implemented in FLAMEGPU_Kernels */

/** add_state_message
 * Function for all types of message partitioning
 * Adds a new state agent to the xmachine_memory_state_list list using a linear mapping
 * @param agents	xmachine_memory_state_list agent list
 * @param state	message variable of type int
 * @param x	message variable of type int
 * @param y	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_state_message(xmachine_message_state_list* state_messages, int state, int x, int y);
 
/** get_first_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param state_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_state * get_first_state_message(xmachine_message_state_list* state_messages, int agentx, int agent_y);

/** get_next_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param state_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_state * get_next_state_message(xmachine_message_state* current, xmachine_message_state_list* state_messages);
  
  
  
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
 * @param h_cells Pointer to agent list on the host
 * @param d_cells Pointer to agent list on the GPU device
 * @param h_xmachine_memory_cell_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_cell_list* h_cells_default, xmachine_memory_cell_list* d_cells_default, int h_xmachine_memory_cell_default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_cells Pointer to agent list on the host
 * @param h_xmachine_memory_cell_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_cell_list* h_cells, int* h_xmachine_memory_cell_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_cell_MAX_count
 * Gets the max agent count for the cell agent type 
 * @return		the maximum cell agent count
 */
extern int get_agent_cell_MAX_count();



/** get_agent_cell_default_count
 * Gets the agent count for the cell agent type in state default
 * @return		the current cell agent count in state default
 */
extern int get_agent_cell_default_count();

/** reset_default_count
 * Resets the agent count of the cell in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_cell_default_count();

/** get_device_cell_default_agents
 * Gets a pointer to xmachine_memory_cell_list on the GPU device
 * @return		a xmachine_memory_cell_list on the GPU device
 */
extern xmachine_memory_cell_list* get_device_cell_default_agents();

/** get_host_cell_default_agents
 * Gets a pointer to xmachine_memory_cell_list on the CPU host
 * @return		a xmachine_memory_cell_list on the CPU host
 */
extern xmachine_memory_cell_list* get_host_cell_default_agents();


/** get_cell_population_width
 * Gets an int value representing the xmachine_memory_cell population width.
 * @return		xmachine_memory_cell population width
 */
extern int get_cell_population_width();


/* Host based access of agent variables*/

/** int get_cell_default_variable_state(unsigned int index)
 * Gets the value of the state variable of an cell agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable state
 */
__host__ int get_cell_default_variable_state(unsigned int index);

/** int get_cell_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an cell agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_cell_default_variable_x(unsigned int index);

/** int get_cell_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an cell agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_cell_default_variable_y(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_cell
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated cell struct.
 */
xmachine_memory_cell* h_allocate_agent_cell();
/** h_free_agent_cell
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_cell(xmachine_memory_cell** agent);
/** h_allocate_agent_cell_array
 * Utility function to allocate an array of structs for  cell agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_cell** h_allocate_agent_cell_array(unsigned int count);
/** h_free_agent_cell_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_cell_array(xmachine_memory_cell*** agents, unsigned int count);


/** h_add_agent_cell_default
 * Host function to add a single agent of type cell to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_cell_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_cell_default(xmachine_memory_cell* agent);

/** h_add_agents_cell_default(
 * Host function to add multiple agents of type cell to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of cell agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_cell_default(xmachine_memory_cell** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_cell_default_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_cell_default_state_variable();

/** int count_cell_default_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_cell_default_state_variable(int count_value);

/** int reduce_cell_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_cell_default_x_variable();

/** int count_cell_default_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_cell_default_x_variable(int count_value);

/** int reduce_cell_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_cell_default_y_variable();

/** int count_cell_default_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_cell_default_y_variable(int count_value);


  
/* global constant variables */


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

#endif //__HEADER

