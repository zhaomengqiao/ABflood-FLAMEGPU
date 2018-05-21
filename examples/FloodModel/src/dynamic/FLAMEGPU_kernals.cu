

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

#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */

__constant__ int d_xmachine_memory_FloodCell_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_FloodCell_Default_count;


/* Message constants */

/* WetDryMessage Message variables */
//Discrete Partitioning Variables
__constant__ int d_message_WetDryMessage_range;     /**< range of the discrete message*/
__constant__ int d_message_WetDryMessage_width;     /**< with of the message grid*/

/* SpaceOperatorMessage Message variables */
//Discrete Partitioning Variables
__constant__ int d_message_SpaceOperatorMessage_range;     /**< range of the discrete message*/
__constant__ int d_message_SpaceOperatorMessage_width;     /**< with of the message grid*/

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* WetDryMessage Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_WetDryMessage_inDomain;
__constant__ int d_tex_xmachine_message_WetDryMessage_inDomain_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_WetDryMessage_x;
__constant__ int d_tex_xmachine_message_WetDryMessage_x_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_WetDryMessage_y;
__constant__ int d_tex_xmachine_message_WetDryMessage_y_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_WetDryMessage_min_hloc;
__constant__ int d_tex_xmachine_message_WetDryMessage_min_hloc_offset;

/* SpaceOperatorMessage Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_inDomain;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_x;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_x_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_y;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_y_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_hFace_E;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_etFace_E;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_X_E;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_hFace_W;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_etFace_W;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_X_W;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_hFace_N;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_etFace_N;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_X_N;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_hFace_S;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_etFace_S;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_X_S;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset;texture<int2, 1, cudaReadModeElementType> tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S;
__constant__ int d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset;

    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ bool next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ bool next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created FloodCell agent functions */

/** reset_FloodCell_scan_input
 * FloodCell agent reset scan input function
 * @param agents The xmachine_memory_FloodCell_list agent list
 */
__global__ void reset_FloodCell_scan_input(xmachine_memory_FloodCell_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created WetDryMessage message functions */


/* Message functions */

template <int AGENT_TYPE>
__device__ void add_WetDryMessage_message(xmachine_message_WetDryMessage_list* messages, int inDomain, int x, int y, double min_hloc){
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int index = global_position.x + (global_position.y * width);

		
		messages->inDomain[index] = inDomain;			
		messages->x[index] = x;			
		messages->y[index] = y;			
		messages->min_hloc[index] = min_hloc;			
	}
	//else CONTINUOUS agents can not write to discrete space
}

//Used by continuous agents this accesses messages with texture cache. agent_x and agent_y are discrete positions in the message space
__device__ xmachine_message_WetDryMessage* get_first_WetDryMessage_message_continuous(xmachine_message_WetDryMessage_list* messages,  int agent_x, int agent_y){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_WetDryMessage* message_share = (xmachine_message_WetDryMessage*)&sm_data[0];
	
	int range = d_message_WetDryMessage_range;
	int width = d_message_WetDryMessage_width;
	
	glm::ivec2 global_position;
	global_position.x = sWRAP(agent_x-range , width);
	global_position.y = sWRAP(agent_y-range , width);
	

	int index = ((global_position.y)* width) + global_position.x;
	
	xmachine_message_WetDryMessage temp_message;
	temp_message._position = glm::ivec2(agent_x, agent_y);
	temp_message._relative = glm::ivec2(-range, -range);

	temp_message.inDomain = tex1Dfetch(tex_xmachine_message_WetDryMessage_inDomain, index + d_tex_xmachine_message_WetDryMessage_inDomain_offset);temp_message.x = tex1Dfetch(tex_xmachine_message_WetDryMessage_x, index + d_tex_xmachine_message_WetDryMessage_x_offset);temp_message.y = tex1Dfetch(tex_xmachine_message_WetDryMessage_y, index + d_tex_xmachine_message_WetDryMessage_y_offset);temp_message.min_hloc = tex1DfetchDouble(tex_xmachine_message_WetDryMessage_min_hloc, index + d_tex_xmachine_message_WetDryMessage_min_hloc_offset);
  
	
	message_share[threadIdx.x] = temp_message;

	//return top left of messages
	return &message_share[threadIdx.x];
}

//Get next WetDryMessage message  continuous
//Used by continuous agents this accesses messages with texture cache (agent position in discrete space was set when accessing first message)
__device__ xmachine_message_WetDryMessage* get_next_WetDryMessage_message_continuous(xmachine_message_WetDryMessage* message, xmachine_message_WetDryMessage_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_WetDryMessage* message_share = (xmachine_message_WetDryMessage*)&sm_data[0];
	
	int range = d_message_WetDryMessage_range;
	int width = d_message_WetDryMessage_width;

	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == (range))
        if (previous_relative.y == (range))
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;

	glm::ivec2 global_position;
	global_position.x =	sWRAP(message->_position.x + next_relative.x, width);
	global_position.y = sWRAP(message->_position.y + next_relative.y, width);

	int index = ((global_position.y)* width) + (global_position.x);
	
	xmachine_message_WetDryMessage temp_message;
	temp_message._position = message->_position;
	temp_message._relative = next_relative;

	temp_message.inDomain = tex1Dfetch(tex_xmachine_message_WetDryMessage_inDomain, index + d_tex_xmachine_message_WetDryMessage_inDomain_offset);	temp_message.x = tex1Dfetch(tex_xmachine_message_WetDryMessage_x, index + d_tex_xmachine_message_WetDryMessage_x_offset);	temp_message.y = tex1Dfetch(tex_xmachine_message_WetDryMessage_y, index + d_tex_xmachine_message_WetDryMessage_y_offset);	temp_message.min_hloc = tex1DfetchDouble(tex_xmachine_message_WetDryMessage_min_hloc, index + d_tex_xmachine_message_WetDryMessage_min_hloc_offset);

	message_share[threadIdx.x] = temp_message;

	return &message_share[threadIdx.x];
}

//method used by discrete agents accessing discrete messages to load messages into shared memory
__device__ void WetDryMessage_message_to_sm(xmachine_message_WetDryMessage_list* messages, char* message_share, int sm_index, int global_index){
		xmachine_message_WetDryMessage temp_message;
		
		temp_message.inDomain = messages->inDomain[global_index];		
		temp_message.x = messages->x[global_index];		
		temp_message.y = messages->y[global_index];		
		temp_message.min_hloc = messages->min_hloc[global_index];		

	  int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_WetDryMessage));
	  xmachine_message_WetDryMessage* sm_message = ((xmachine_message_WetDryMessage*)&message_share[message_index]);
	  sm_message[0] = temp_message;
}

//Get first WetDryMessage message 
//Used by discrete agents this accesses messages with texture cache. Agent position is determined by position in the grid/block
//Possibility of upto 8 thread divergences
__device__ xmachine_message_WetDryMessage* get_first_WetDryMessage_message_discrete(xmachine_message_WetDryMessage_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();

	int range = d_message_WetDryMessage_range;
	int width = d_message_WetDryMessage_width;
	int sm_grid_width = blockDim.x + (range* 2);
	
	
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//calculate the position in shared memory of first load
	glm::ivec2 sm_pos;
	sm_pos.x = threadIdx.x + range;
	sm_pos.y = threadIdx.y + range;
	int sm_index = (sm_pos.y * sm_grid_width) + sm_pos.x;

	//each thread loads to shared memory (coalesced read)
	WetDryMessage_message_to_sm(messages, message_share, sm_index, index);

	//check for edge conditions
	int left_border = (threadIdx.x < range);
	int right_border = (threadIdx.x >= (blockDim.x-range));
	int top_border = (threadIdx.y < range);
	int bottom_border = (threadIdx.y >= (blockDim.y-range));

	
	int  border_index;
	int  sm_border_index;

	//left
	if (left_border){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + threadIdx.x;
		
		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//right
	if (right_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + (sm_pos.x + range);

		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top
	if (top_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + sm_pos.x;

		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom
	if (bottom_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + sm_pos.x;

		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top left
	if ((top_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
		
		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top right
	if ((top_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + (sm_pos.x + range);
		
		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom right
	if ((bottom_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + (sm_pos.x + range);
		
		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom left
	if ((bottom_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + threadIdx.x;
		
		WetDryMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	__syncthreads();
	
  
	//top left of block position sm index
	sm_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
	
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_WetDryMessage));
	xmachine_message_WetDryMessage* temp = ((xmachine_message_WetDryMessage*)&message_share[message_index]);
	temp->_relative = glm::ivec2(-range, -range); //this is the relative position
	return temp;
}

//Get next WetDryMessage message 
//Used by discrete agents this accesses messages through shared memory which were all loaded on first message retrieval call.
__device__ xmachine_message_WetDryMessage* get_next_WetDryMessage_message_discrete(xmachine_message_WetDryMessage* message, xmachine_message_WetDryMessage_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();
	
	int range = d_message_WetDryMessage_range;
	int sm_grid_width = blockDim.x+(range*2);


	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == range)
        if (previous_relative.y == range)
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;


	//calculate the next message position
	glm::ivec2 next_position;// = block_position+next_relative;
	//offset next position by the sm border size
	next_position.x = threadIdx.x + next_relative.x + range;
	next_position.y = threadIdx.y + next_relative.y + range;

	int sm_index = next_position.x + (next_position.y * sm_grid_width);
	
	__syncthreads();
  
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_WetDryMessage));
	xmachine_message_WetDryMessage* temp = ((xmachine_message_WetDryMessage*)&message_share[message_index]);
	temp->_relative = next_relative; //this is the relative position
	return temp;
}

//Get first WetDryMessage message
template <int AGENT_TYPE>
__device__ xmachine_message_WetDryMessage* get_first_WetDryMessage_message(xmachine_message_WetDryMessage_list* messages, int agent_x, int agent_y){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_first_WetDryMessage_message_discrete(messages);
	else	//use texture fetching method
		return get_first_WetDryMessage_message_continuous(messages, agent_x, agent_y);

}

//Get next WetDryMessage message
template <int AGENT_TYPE>
__device__ xmachine_message_WetDryMessage* get_next_WetDryMessage_message(xmachine_message_WetDryMessage* message, xmachine_message_WetDryMessage_list* messages){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_next_WetDryMessage_message_discrete(message, messages);
	else	//use texture fetching method
		return get_next_WetDryMessage_message_continuous(message, messages);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created SpaceOperatorMessage message functions */


/* Message functions */

template <int AGENT_TYPE>
__device__ void add_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage_list* messages, int inDomain, int x, int y, double hFace_E, double etFace_E, double qFace_X_E, double qFace_Y_E, double hFace_W, double etFace_W, double qFace_X_W, double qFace_Y_W, double hFace_N, double etFace_N, double qFace_X_N, double qFace_Y_N, double hFace_S, double etFace_S, double qFace_X_S, double qFace_Y_S){
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int index = global_position.x + (global_position.y * width);

		
		messages->inDomain[index] = inDomain;			
		messages->x[index] = x;			
		messages->y[index] = y;			
		messages->hFace_E[index] = hFace_E;			
		messages->etFace_E[index] = etFace_E;			
		messages->qFace_X_E[index] = qFace_X_E;			
		messages->qFace_Y_E[index] = qFace_Y_E;			
		messages->hFace_W[index] = hFace_W;			
		messages->etFace_W[index] = etFace_W;			
		messages->qFace_X_W[index] = qFace_X_W;			
		messages->qFace_Y_W[index] = qFace_Y_W;			
		messages->hFace_N[index] = hFace_N;			
		messages->etFace_N[index] = etFace_N;			
		messages->qFace_X_N[index] = qFace_X_N;			
		messages->qFace_Y_N[index] = qFace_Y_N;			
		messages->hFace_S[index] = hFace_S;			
		messages->etFace_S[index] = etFace_S;			
		messages->qFace_X_S[index] = qFace_X_S;			
		messages->qFace_Y_S[index] = qFace_Y_S;			
	}
	//else CONTINUOUS agents can not write to discrete space
}

//Used by continuous agents this accesses messages with texture cache. agent_x and agent_y are discrete positions in the message space
__device__ xmachine_message_SpaceOperatorMessage* get_first_SpaceOperatorMessage_message_continuous(xmachine_message_SpaceOperatorMessage_list* messages,  int agent_x, int agent_y){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_SpaceOperatorMessage* message_share = (xmachine_message_SpaceOperatorMessage*)&sm_data[0];
	
	int range = d_message_SpaceOperatorMessage_range;
	int width = d_message_SpaceOperatorMessage_width;
	
	glm::ivec2 global_position;
	global_position.x = sWRAP(agent_x-range , width);
	global_position.y = sWRAP(agent_y-range , width);
	

	int index = ((global_position.y)* width) + global_position.x;
	
	xmachine_message_SpaceOperatorMessage temp_message;
	temp_message._position = glm::ivec2(agent_x, agent_y);
	temp_message._relative = glm::ivec2(-range, -range);

	temp_message.inDomain = tex1Dfetch(tex_xmachine_message_SpaceOperatorMessage_inDomain, index + d_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset);temp_message.x = tex1Dfetch(tex_xmachine_message_SpaceOperatorMessage_x, index + d_tex_xmachine_message_SpaceOperatorMessage_x_offset);temp_message.y = tex1Dfetch(tex_xmachine_message_SpaceOperatorMessage_y, index + d_tex_xmachine_message_SpaceOperatorMessage_y_offset);temp_message.hFace_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_E, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset);
  temp_message.etFace_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_E, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset);
  temp_message.qFace_X_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_E, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset);
  temp_message.qFace_Y_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset);
  temp_message.hFace_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_W, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset);
  temp_message.etFace_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_W, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset);
  temp_message.qFace_X_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_W, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset);
  temp_message.qFace_Y_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset);
  temp_message.hFace_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_N, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset);
  temp_message.etFace_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_N, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset);
  temp_message.qFace_X_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_N, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset);
  temp_message.qFace_Y_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset);
  temp_message.hFace_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_S, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset);
  temp_message.etFace_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_S, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset);
  temp_message.qFace_X_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_S, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset);
  temp_message.qFace_Y_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset);
  
	
	message_share[threadIdx.x] = temp_message;

	//return top left of messages
	return &message_share[threadIdx.x];
}

//Get next SpaceOperatorMessage message  continuous
//Used by continuous agents this accesses messages with texture cache (agent position in discrete space was set when accessing first message)
__device__ xmachine_message_SpaceOperatorMessage* get_next_SpaceOperatorMessage_message_continuous(xmachine_message_SpaceOperatorMessage* message, xmachine_message_SpaceOperatorMessage_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_SpaceOperatorMessage* message_share = (xmachine_message_SpaceOperatorMessage*)&sm_data[0];
	
	int range = d_message_SpaceOperatorMessage_range;
	int width = d_message_SpaceOperatorMessage_width;

	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == (range))
        if (previous_relative.y == (range))
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;

	glm::ivec2 global_position;
	global_position.x =	sWRAP(message->_position.x + next_relative.x, width);
	global_position.y = sWRAP(message->_position.y + next_relative.y, width);

	int index = ((global_position.y)* width) + (global_position.x);
	
	xmachine_message_SpaceOperatorMessage temp_message;
	temp_message._position = message->_position;
	temp_message._relative = next_relative;

	temp_message.inDomain = tex1Dfetch(tex_xmachine_message_SpaceOperatorMessage_inDomain, index + d_tex_xmachine_message_SpaceOperatorMessage_inDomain_offset);	temp_message.x = tex1Dfetch(tex_xmachine_message_SpaceOperatorMessage_x, index + d_tex_xmachine_message_SpaceOperatorMessage_x_offset);	temp_message.y = tex1Dfetch(tex_xmachine_message_SpaceOperatorMessage_y, index + d_tex_xmachine_message_SpaceOperatorMessage_y_offset);	temp_message.hFace_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_E, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_E_offset);temp_message.etFace_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_E, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_E_offset);temp_message.qFace_X_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_E, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_E_offset);temp_message.qFace_Y_E = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_E_offset);temp_message.hFace_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_W, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_W_offset);temp_message.etFace_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_W, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_W_offset);temp_message.qFace_X_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_W, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_W_offset);temp_message.qFace_Y_W = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_W_offset);temp_message.hFace_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_N, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_N_offset);temp_message.etFace_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_N, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_N_offset);temp_message.qFace_X_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_N, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_N_offset);temp_message.qFace_Y_N = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_N_offset);temp_message.hFace_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_hFace_S, index + d_tex_xmachine_message_SpaceOperatorMessage_hFace_S_offset);temp_message.etFace_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_etFace_S, index + d_tex_xmachine_message_SpaceOperatorMessage_etFace_S_offset);temp_message.qFace_X_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_X_S, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_X_S_offset);temp_message.qFace_Y_S = tex1DfetchDouble(tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S, index + d_tex_xmachine_message_SpaceOperatorMessage_qFace_Y_S_offset);

	message_share[threadIdx.x] = temp_message;

	return &message_share[threadIdx.x];
}

//method used by discrete agents accessing discrete messages to load messages into shared memory
__device__ void SpaceOperatorMessage_message_to_sm(xmachine_message_SpaceOperatorMessage_list* messages, char* message_share, int sm_index, int global_index){
		xmachine_message_SpaceOperatorMessage temp_message;
		
		temp_message.inDomain = messages->inDomain[global_index];		
		temp_message.x = messages->x[global_index];		
		temp_message.y = messages->y[global_index];		
		temp_message.hFace_E = messages->hFace_E[global_index];		
		temp_message.etFace_E = messages->etFace_E[global_index];		
		temp_message.qFace_X_E = messages->qFace_X_E[global_index];		
		temp_message.qFace_Y_E = messages->qFace_Y_E[global_index];		
		temp_message.hFace_W = messages->hFace_W[global_index];		
		temp_message.etFace_W = messages->etFace_W[global_index];		
		temp_message.qFace_X_W = messages->qFace_X_W[global_index];		
		temp_message.qFace_Y_W = messages->qFace_Y_W[global_index];		
		temp_message.hFace_N = messages->hFace_N[global_index];		
		temp_message.etFace_N = messages->etFace_N[global_index];		
		temp_message.qFace_X_N = messages->qFace_X_N[global_index];		
		temp_message.qFace_Y_N = messages->qFace_Y_N[global_index];		
		temp_message.hFace_S = messages->hFace_S[global_index];		
		temp_message.etFace_S = messages->etFace_S[global_index];		
		temp_message.qFace_X_S = messages->qFace_X_S[global_index];		
		temp_message.qFace_Y_S = messages->qFace_Y_S[global_index];		

	  int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_SpaceOperatorMessage));
	  xmachine_message_SpaceOperatorMessage* sm_message = ((xmachine_message_SpaceOperatorMessage*)&message_share[message_index]);
	  sm_message[0] = temp_message;
}

//Get first SpaceOperatorMessage message 
//Used by discrete agents this accesses messages with texture cache. Agent position is determined by position in the grid/block
//Possibility of upto 8 thread divergences
__device__ xmachine_message_SpaceOperatorMessage* get_first_SpaceOperatorMessage_message_discrete(xmachine_message_SpaceOperatorMessage_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();

	int range = d_message_SpaceOperatorMessage_range;
	int width = d_message_SpaceOperatorMessage_width;
	int sm_grid_width = blockDim.x + (range* 2);
	
	
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//calculate the position in shared memory of first load
	glm::ivec2 sm_pos;
	sm_pos.x = threadIdx.x + range;
	sm_pos.y = threadIdx.y + range;
	int sm_index = (sm_pos.y * sm_grid_width) + sm_pos.x;

	//each thread loads to shared memory (coalesced read)
	SpaceOperatorMessage_message_to_sm(messages, message_share, sm_index, index);

	//check for edge conditions
	int left_border = (threadIdx.x < range);
	int right_border = (threadIdx.x >= (blockDim.x-range));
	int top_border = (threadIdx.y < range);
	int bottom_border = (threadIdx.y >= (blockDim.y-range));

	
	int  border_index;
	int  sm_border_index;

	//left
	if (left_border){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + threadIdx.x;
		
		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//right
	if (right_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + (sm_pos.x + range);

		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top
	if (top_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + sm_pos.x;

		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom
	if (bottom_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + sm_pos.x;

		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top left
	if ((top_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
		
		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top right
	if ((top_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + (sm_pos.x + range);
		
		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom right
	if ((bottom_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + (sm_pos.x + range);
		
		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom left
	if ((bottom_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + threadIdx.x;
		
		SpaceOperatorMessage_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	__syncthreads();
	
  
	//top left of block position sm index
	sm_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
	
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_SpaceOperatorMessage));
	xmachine_message_SpaceOperatorMessage* temp = ((xmachine_message_SpaceOperatorMessage*)&message_share[message_index]);
	temp->_relative = glm::ivec2(-range, -range); //this is the relative position
	return temp;
}

//Get next SpaceOperatorMessage message 
//Used by discrete agents this accesses messages through shared memory which were all loaded on first message retrieval call.
__device__ xmachine_message_SpaceOperatorMessage* get_next_SpaceOperatorMessage_message_discrete(xmachine_message_SpaceOperatorMessage* message, xmachine_message_SpaceOperatorMessage_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();
	
	int range = d_message_SpaceOperatorMessage_range;
	int sm_grid_width = blockDim.x+(range*2);


	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == range)
        if (previous_relative.y == range)
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;


	//calculate the next message position
	glm::ivec2 next_position;// = block_position+next_relative;
	//offset next position by the sm border size
	next_position.x = threadIdx.x + next_relative.x + range;
	next_position.y = threadIdx.y + next_relative.y + range;

	int sm_index = next_position.x + (next_position.y * sm_grid_width);
	
	__syncthreads();
  
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_SpaceOperatorMessage));
	xmachine_message_SpaceOperatorMessage* temp = ((xmachine_message_SpaceOperatorMessage*)&message_share[message_index]);
	temp->_relative = next_relative; //this is the relative position
	return temp;
}

//Get first SpaceOperatorMessage message
template <int AGENT_TYPE>
__device__ xmachine_message_SpaceOperatorMessage* get_first_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage_list* messages, int agent_x, int agent_y){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_first_SpaceOperatorMessage_message_discrete(messages);
	else	//use texture fetching method
		return get_first_SpaceOperatorMessage_message_continuous(messages, agent_x, agent_y);

}

//Get next SpaceOperatorMessage message
template <int AGENT_TYPE>
__device__ xmachine_message_SpaceOperatorMessage* get_next_SpaceOperatorMessage_message(xmachine_message_SpaceOperatorMessage* message, xmachine_message_SpaceOperatorMessage_list* messages){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_next_SpaceOperatorMessage_message_discrete(message, messages);
	else	//use texture fetching method
		return get_next_SpaceOperatorMessage_message_continuous(message, messages);

}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_PrepareWetDry(xmachine_memory_FloodCell_list* agents, xmachine_message_WetDryMessage_list* WetDryMessage_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_PrepareWetDry Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_FloodCell agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.inDomain = agents->inDomain[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z0 = agents->z0[index];
	agent.h = agents->h[index];
	agent.qx = agents->qx[index];
	agent.qy = agents->qy[index];
	agent.timeStep = agents->timeStep[index];
	agent.minh_loc = agents->minh_loc[index];
	agent.hFace_E = agents->hFace_E[index];
	agent.etFace_E = agents->etFace_E[index];
	agent.qxFace_E = agents->qxFace_E[index];
	agent.qyFace_E = agents->qyFace_E[index];
	agent.hFace_W = agents->hFace_W[index];
	agent.etFace_W = agents->etFace_W[index];
	agent.qxFace_W = agents->qxFace_W[index];
	agent.qyFace_W = agents->qyFace_W[index];
	agent.hFace_N = agents->hFace_N[index];
	agent.etFace_N = agents->etFace_N[index];
	agent.qxFace_N = agents->qxFace_N[index];
	agent.qyFace_N = agents->qyFace_N[index];
	agent.hFace_S = agents->hFace_S[index];
	agent.etFace_S = agents->etFace_S[index];
	agent.qxFace_S = agents->qxFace_S[index];
	agent.qyFace_S = agents->qyFace_S[index];

	//FLAME function call
	PrepareWetDry(&agent, WetDryMessage_messages	);
	

	

	//AoS to SoA - xmachine_memory_PrepareWetDry Coalesced memory write (ignore arrays)
	agents->inDomain[index] = agent.inDomain;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z0[index] = agent.z0;
	agents->h[index] = agent.h;
	agents->qx[index] = agent.qx;
	agents->qy[index] = agent.qy;
	agents->timeStep[index] = agent.timeStep;
	agents->minh_loc[index] = agent.minh_loc;
	agents->hFace_E[index] = agent.hFace_E;
	agents->etFace_E[index] = agent.etFace_E;
	agents->qxFace_E[index] = agent.qxFace_E;
	agents->qyFace_E[index] = agent.qyFace_E;
	agents->hFace_W[index] = agent.hFace_W;
	agents->etFace_W[index] = agent.etFace_W;
	agents->qxFace_W[index] = agent.qxFace_W;
	agents->qyFace_W[index] = agent.qyFace_W;
	agents->hFace_N[index] = agent.hFace_N;
	agents->etFace_N[index] = agent.etFace_N;
	agents->qxFace_N[index] = agent.qxFace_N;
	agents->qyFace_N[index] = agent.qyFace_N;
	agents->hFace_S[index] = agent.hFace_S;
	agents->etFace_S[index] = agent.etFace_S;
	agents->qxFace_S[index] = agent.qxFace_S;
	agents->qyFace_S[index] = agent.qyFace_S;
}

/**
 *
 */
__global__ void GPUFLAME_ProcessWetDryMessage(xmachine_memory_FloodCell_list* agents, xmachine_message_WetDryMessage_list* WetDryMessage_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_ProcessWetDryMessage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_FloodCell agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.inDomain = agents->inDomain[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z0 = agents->z0[index];
	agent.h = agents->h[index];
	agent.qx = agents->qx[index];
	agent.qy = agents->qy[index];
	agent.timeStep = agents->timeStep[index];
	agent.minh_loc = agents->minh_loc[index];
	agent.hFace_E = agents->hFace_E[index];
	agent.etFace_E = agents->etFace_E[index];
	agent.qxFace_E = agents->qxFace_E[index];
	agent.qyFace_E = agents->qyFace_E[index];
	agent.hFace_W = agents->hFace_W[index];
	agent.etFace_W = agents->etFace_W[index];
	agent.qxFace_W = agents->qxFace_W[index];
	agent.qyFace_W = agents->qyFace_W[index];
	agent.hFace_N = agents->hFace_N[index];
	agent.etFace_N = agents->etFace_N[index];
	agent.qxFace_N = agents->qxFace_N[index];
	agent.qyFace_N = agents->qyFace_N[index];
	agent.hFace_S = agents->hFace_S[index];
	agent.etFace_S = agents->etFace_S[index];
	agent.qxFace_S = agents->qxFace_S[index];
	agent.qyFace_S = agents->qyFace_S[index];

	//FLAME function call
	ProcessWetDryMessage(&agent, WetDryMessage_messages);
	

	

	//AoS to SoA - xmachine_memory_ProcessWetDryMessage Coalesced memory write (ignore arrays)
	agents->inDomain[index] = agent.inDomain;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z0[index] = agent.z0;
	agents->h[index] = agent.h;
	agents->qx[index] = agent.qx;
	agents->qy[index] = agent.qy;
	agents->timeStep[index] = agent.timeStep;
	agents->minh_loc[index] = agent.minh_loc;
	agents->hFace_E[index] = agent.hFace_E;
	agents->etFace_E[index] = agent.etFace_E;
	agents->qxFace_E[index] = agent.qxFace_E;
	agents->qyFace_E[index] = agent.qyFace_E;
	agents->hFace_W[index] = agent.hFace_W;
	agents->etFace_W[index] = agent.etFace_W;
	agents->qxFace_W[index] = agent.qxFace_W;
	agents->qyFace_W[index] = agent.qyFace_W;
	agents->hFace_N[index] = agent.hFace_N;
	agents->etFace_N[index] = agent.etFace_N;
	agents->qxFace_N[index] = agent.qxFace_N;
	agents->qyFace_N[index] = agent.qyFace_N;
	agents->hFace_S[index] = agent.hFace_S;
	agents->etFace_S[index] = agent.etFace_S;
	agents->qxFace_S[index] = agent.qxFace_S;
	agents->qyFace_S[index] = agent.qyFace_S;
}

/**
 *
 */
__global__ void GPUFLAME_PrepareSpaceOperator(xmachine_memory_FloodCell_list* agents, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_PrepareSpaceOperator Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_FloodCell agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.inDomain = agents->inDomain[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z0 = agents->z0[index];
	agent.h = agents->h[index];
	agent.qx = agents->qx[index];
	agent.qy = agents->qy[index];
	agent.timeStep = agents->timeStep[index];
	agent.minh_loc = agents->minh_loc[index];
	agent.hFace_E = agents->hFace_E[index];
	agent.etFace_E = agents->etFace_E[index];
	agent.qxFace_E = agents->qxFace_E[index];
	agent.qyFace_E = agents->qyFace_E[index];
	agent.hFace_W = agents->hFace_W[index];
	agent.etFace_W = agents->etFace_W[index];
	agent.qxFace_W = agents->qxFace_W[index];
	agent.qyFace_W = agents->qyFace_W[index];
	agent.hFace_N = agents->hFace_N[index];
	agent.etFace_N = agents->etFace_N[index];
	agent.qxFace_N = agents->qxFace_N[index];
	agent.qyFace_N = agents->qyFace_N[index];
	agent.hFace_S = agents->hFace_S[index];
	agent.etFace_S = agents->etFace_S[index];
	agent.qxFace_S = agents->qxFace_S[index];
	agent.qyFace_S = agents->qyFace_S[index];

	//FLAME function call
	PrepareSpaceOperator(&agent, SpaceOperatorMessage_messages	);
	

	

	//AoS to SoA - xmachine_memory_PrepareSpaceOperator Coalesced memory write (ignore arrays)
	agents->inDomain[index] = agent.inDomain;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z0[index] = agent.z0;
	agents->h[index] = agent.h;
	agents->qx[index] = agent.qx;
	agents->qy[index] = agent.qy;
	agents->timeStep[index] = agent.timeStep;
	agents->minh_loc[index] = agent.minh_loc;
	agents->hFace_E[index] = agent.hFace_E;
	agents->etFace_E[index] = agent.etFace_E;
	agents->qxFace_E[index] = agent.qxFace_E;
	agents->qyFace_E[index] = agent.qyFace_E;
	agents->hFace_W[index] = agent.hFace_W;
	agents->etFace_W[index] = agent.etFace_W;
	agents->qxFace_W[index] = agent.qxFace_W;
	agents->qyFace_W[index] = agent.qyFace_W;
	agents->hFace_N[index] = agent.hFace_N;
	agents->etFace_N[index] = agent.etFace_N;
	agents->qxFace_N[index] = agent.qxFace_N;
	agents->qyFace_N[index] = agent.qyFace_N;
	agents->hFace_S[index] = agent.hFace_S;
	agents->etFace_S[index] = agent.etFace_S;
	agents->qxFace_S[index] = agent.qxFace_S;
	agents->qyFace_S[index] = agent.qyFace_S;
}

/**
 *
 */
__global__ void GPUFLAME_ProcessSpaceOperatorMessage(xmachine_memory_FloodCell_list* agents, xmachine_message_SpaceOperatorMessage_list* SpaceOperatorMessage_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_ProcessSpaceOperatorMessage Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_FloodCell agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.inDomain = agents->inDomain[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z0 = agents->z0[index];
	agent.h = agents->h[index];
	agent.qx = agents->qx[index];
	agent.qy = agents->qy[index];
	agent.timeStep = agents->timeStep[index];
	agent.minh_loc = agents->minh_loc[index];
	agent.hFace_E = agents->hFace_E[index];
	agent.etFace_E = agents->etFace_E[index];
	agent.qxFace_E = agents->qxFace_E[index];
	agent.qyFace_E = agents->qyFace_E[index];
	agent.hFace_W = agents->hFace_W[index];
	agent.etFace_W = agents->etFace_W[index];
	agent.qxFace_W = agents->qxFace_W[index];
	agent.qyFace_W = agents->qyFace_W[index];
	agent.hFace_N = agents->hFace_N[index];
	agent.etFace_N = agents->etFace_N[index];
	agent.qxFace_N = agents->qxFace_N[index];
	agent.qyFace_N = agents->qyFace_N[index];
	agent.hFace_S = agents->hFace_S[index];
	agent.etFace_S = agents->etFace_S[index];
	agent.qxFace_S = agents->qxFace_S[index];
	agent.qyFace_S = agents->qyFace_S[index];

	//FLAME function call
	ProcessSpaceOperatorMessage(&agent, SpaceOperatorMessage_messages);
	

	

	//AoS to SoA - xmachine_memory_ProcessSpaceOperatorMessage Coalesced memory write (ignore arrays)
	agents->inDomain[index] = agent.inDomain;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z0[index] = agent.z0;
	agents->h[index] = agent.h;
	agents->qx[index] = agent.qx;
	agents->qy[index] = agent.qy;
	agents->timeStep[index] = agent.timeStep;
	agents->minh_loc[index] = agent.minh_loc;
	agents->hFace_E[index] = agent.hFace_E;
	agents->etFace_E[index] = agent.etFace_E;
	agents->qxFace_E[index] = agent.qxFace_E;
	agents->qyFace_E[index] = agent.qyFace_E;
	agents->hFace_W[index] = agent.hFace_W;
	agents->etFace_W[index] = agent.etFace_W;
	agents->qxFace_W[index] = agent.qxFace_W;
	agents->qyFace_W[index] = agent.qyFace_W;
	agents->hFace_N[index] = agent.hFace_N;
	agents->etFace_N[index] = agent.etFace_N;
	agents->qxFace_N[index] = agent.qxFace_N;
	agents->qyFace_N[index] = agent.qyFace_N;
	agents->hFace_S[index] = agent.hFace_S;
	agents->etFace_S[index] = agent.etFace_S;
	agents->qxFace_S[index] = agent.qxFace_S;
	agents->qyFace_S[index] = agent.qyFace_S;
}

	
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
