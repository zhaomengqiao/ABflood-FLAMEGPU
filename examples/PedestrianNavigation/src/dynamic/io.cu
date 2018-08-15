
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


#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <algorithm>
#include <string>
#include <vector>



#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template <class T>
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: variable array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template <class T, class BASE_T, unsigned int D>
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput<BASE_T>(parseFunc, token, (BASE_T*) &vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_agents_default, d_agents_default, sizeof(xmachine_memory_agent_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying agent Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_navmaps_static, d_navmaps_static, sizeof(xmachine_memory_navmap_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying navmap Agent static State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("<states>\n<itno>", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("</itno>\n", file);
    fputs("<environment>\n" , file);
    
    fputs("\t<EMMISION_RATE_EXIT1>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT1()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT1>\n", file);
    fputs("\t<EMMISION_RATE_EXIT2>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT2()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT2>\n", file);
    fputs("\t<EMMISION_RATE_EXIT3>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT3()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT3>\n", file);
    fputs("\t<EMMISION_RATE_EXIT4>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT4()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT4>\n", file);
    fputs("\t<EMMISION_RATE_EXIT5>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT5()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT5>\n", file);
    fputs("\t<EMMISION_RATE_EXIT6>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT6()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT6>\n", file);
    fputs("\t<EMMISION_RATE_EXIT7>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT7()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT7>\n", file);
    fputs("\t<EXIT1_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT1_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT1_PROBABILITY>\n", file);
    fputs("\t<EXIT2_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT2_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT2_PROBABILITY>\n", file);
    fputs("\t<EXIT3_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT3_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT3_PROBABILITY>\n", file);
    fputs("\t<EXIT4_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT4_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT4_PROBABILITY>\n", file);
    fputs("\t<EXIT5_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT5_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT5_PROBABILITY>\n", file);
    fputs("\t<EXIT6_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT6_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT6_PROBABILITY>\n", file);
    fputs("\t<EXIT7_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT7_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT7_PROBABILITY>\n", file);
    fputs("\t<EXIT1_STATE>", file);
    sprintf(data, "%d", (*get_EXIT1_STATE()));
    fputs(data, file);
    fputs("</EXIT1_STATE>\n", file);
    fputs("\t<EXIT2_STATE>", file);
    sprintf(data, "%d", (*get_EXIT2_STATE()));
    fputs(data, file);
    fputs("</EXIT2_STATE>\n", file);
    fputs("\t<EXIT3_STATE>", file);
    sprintf(data, "%d", (*get_EXIT3_STATE()));
    fputs(data, file);
    fputs("</EXIT3_STATE>\n", file);
    fputs("\t<EXIT4_STATE>", file);
    sprintf(data, "%d", (*get_EXIT4_STATE()));
    fputs(data, file);
    fputs("</EXIT4_STATE>\n", file);
    fputs("\t<EXIT5_STATE>", file);
    sprintf(data, "%d", (*get_EXIT5_STATE()));
    fputs(data, file);
    fputs("</EXIT5_STATE>\n", file);
    fputs("\t<EXIT6_STATE>", file);
    sprintf(data, "%d", (*get_EXIT6_STATE()));
    fputs(data, file);
    fputs("</EXIT6_STATE>\n", file);
    fputs("\t<EXIT7_STATE>", file);
    sprintf(data, "%d", (*get_EXIT7_STATE()));
    fputs(data, file);
    fputs("</EXIT7_STATE>\n", file);
    fputs("\t<EXIT1_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT1_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT1_CELL_COUNT>\n", file);
    fputs("\t<EXIT2_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT2_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT2_CELL_COUNT>\n", file);
    fputs("\t<EXIT3_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT3_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT3_CELL_COUNT>\n", file);
    fputs("\t<EXIT4_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT4_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT4_CELL_COUNT>\n", file);
    fputs("\t<EXIT5_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT5_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT5_CELL_COUNT>\n", file);
    fputs("\t<EXIT6_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT6_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT6_CELL_COUNT>\n", file);
    fputs("\t<EXIT7_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT7_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT7_CELL_COUNT>\n", file);
    fputs("\t<TIME_SCALER>", file);
    sprintf(data, "%f", (*get_TIME_SCALER()));
    fputs(data, file);
    fputs("</TIME_SCALER>\n", file);
    fputs("\t<STEER_WEIGHT>", file);
    sprintf(data, "%f", (*get_STEER_WEIGHT()));
    fputs(data, file);
    fputs("</STEER_WEIGHT>\n", file);
    fputs("\t<AVOID_WEIGHT>", file);
    sprintf(data, "%f", (*get_AVOID_WEIGHT()));
    fputs(data, file);
    fputs("</AVOID_WEIGHT>\n", file);
    fputs("\t<COLLISION_WEIGHT>", file);
    sprintf(data, "%f", (*get_COLLISION_WEIGHT()));
    fputs(data, file);
    fputs("</COLLISION_WEIGHT>\n", file);
    fputs("\t<GOAL_WEIGHT>", file);
    sprintf(data, "%f", (*get_GOAL_WEIGHT()));
    fputs(data, file);
    fputs("</GOAL_WEIGHT>\n", file);
	fputs("</environment>\n" , file);

	//Write each agent agent to xml
	for (int i=0; i<h_xmachine_memory_agent_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>agent</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_agents_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_agents_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<velx>", file);
        sprintf(data, "%f", h_agents_default->velx[i]);
		fputs(data, file);
		fputs("</velx>\n", file);
        
		fputs("<vely>", file);
        sprintf(data, "%f", h_agents_default->vely[i]);
		fputs(data, file);
		fputs("</vely>\n", file);
        
		fputs("<steer_x>", file);
        sprintf(data, "%f", h_agents_default->steer_x[i]);
		fputs(data, file);
		fputs("</steer_x>\n", file);
        
		fputs("<steer_y>", file);
        sprintf(data, "%f", h_agents_default->steer_y[i]);
		fputs(data, file);
		fputs("</steer_y>\n", file);
        
		fputs("<height>", file);
        sprintf(data, "%f", h_agents_default->height[i]);
		fputs(data, file);
		fputs("</height>\n", file);
        
		fputs("<exit_no>", file);
        sprintf(data, "%d", h_agents_default->exit_no[i]);
		fputs(data, file);
		fputs("</exit_no>\n", file);
        
		fputs("<speed>", file);
        sprintf(data, "%f", h_agents_default->speed[i]);
		fputs(data, file);
		fputs("</speed>\n", file);
        
		fputs("<lod>", file);
        sprintf(data, "%d", h_agents_default->lod[i]);
		fputs(data, file);
		fputs("</lod>\n", file);
        
		fputs("<animate>", file);
        sprintf(data, "%f", h_agents_default->animate[i]);
		fputs(data, file);
		fputs("</animate>\n", file);
        
		fputs("<animate_dir>", file);
        sprintf(data, "%d", h_agents_default->animate_dir[i]);
		fputs(data, file);
		fputs("</animate_dir>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each navmap agent to xml
	for (int i=0; i<h_xmachine_memory_navmap_static_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>navmap</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_navmaps_static->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_navmaps_static->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<exit_no>", file);
        sprintf(data, "%d", h_navmaps_static->exit_no[i]);
		fputs(data, file);
		fputs("</exit_no>\n", file);
        
		fputs("<height>", file);
        sprintf(data, "%f", h_navmaps_static->height[i]);
		fputs(data, file);
		fputs("</height>\n", file);
        
		fputs("<collision_x>", file);
        sprintf(data, "%f", h_navmaps_static->collision_x[i]);
		fputs(data, file);
		fputs("</collision_x>\n", file);
        
		fputs("<collision_y>", file);
        sprintf(data, "%f", h_navmaps_static->collision_y[i]);
		fputs(data, file);
		fputs("</collision_y>\n", file);
        
		fputs("<exit0_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit0_x[i]);
		fputs(data, file);
		fputs("</exit0_x>\n", file);
        
		fputs("<exit0_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit0_y[i]);
		fputs(data, file);
		fputs("</exit0_y>\n", file);
        
		fputs("<exit1_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit1_x[i]);
		fputs(data, file);
		fputs("</exit1_x>\n", file);
        
		fputs("<exit1_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit1_y[i]);
		fputs(data, file);
		fputs("</exit1_y>\n", file);
        
		fputs("<exit2_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit2_x[i]);
		fputs(data, file);
		fputs("</exit2_x>\n", file);
        
		fputs("<exit2_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit2_y[i]);
		fputs(data, file);
		fputs("</exit2_y>\n", file);
        
		fputs("<exit3_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit3_x[i]);
		fputs(data, file);
		fputs("</exit3_x>\n", file);
        
		fputs("<exit3_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit3_y[i]);
		fputs(data, file);
		fputs("</exit3_y>\n", file);
        
		fputs("<exit4_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit4_x[i]);
		fputs(data, file);
		fputs("</exit4_x>\n", file);
        
		fputs("<exit4_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit4_y[i]);
		fputs(data, file);
		fputs("</exit4_y>\n", file);
        
		fputs("<exit5_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit5_x[i]);
		fputs(data, file);
		fputs("</exit5_x>\n", file);
        
		fputs("<exit5_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit5_y[i]);
		fputs(data, file);
		fputs("</exit5_y>\n", file);
        
		fputs("<exit6_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit6_x[i]);
		fputs(data, file);
		fputs("</exit6_x>\n", file);
        
		fputs("<exit6_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit6_y[i]);
		fputs(data, file);
		fputs("</exit6_y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	const int bufferSize = 10000;
	char buffer[bufferSize];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name, in_comment;
    int in_agent_x;
    int in_agent_y;
    int in_agent_velx;
    int in_agent_vely;
    int in_agent_steer_x;
    int in_agent_steer_y;
    int in_agent_height;
    int in_agent_exit_no;
    int in_agent_speed;
    int in_agent_lod;
    int in_agent_animate;
    int in_agent_animate_dir;
    int in_navmap_x;
    int in_navmap_y;
    int in_navmap_exit_no;
    int in_navmap_height;
    int in_navmap_collision_x;
    int in_navmap_collision_y;
    int in_navmap_exit0_x;
    int in_navmap_exit0_y;
    int in_navmap_exit1_x;
    int in_navmap_exit1_y;
    int in_navmap_exit2_x;
    int in_navmap_exit2_y;
    int in_navmap_exit3_x;
    int in_navmap_exit3_y;
    int in_navmap_exit4_x;
    int in_navmap_exit4_y;
    int in_navmap_exit5_x;
    int in_navmap_exit5_y;
    int in_navmap_exit6_x;
    int in_navmap_exit6_y;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_EMMISION_RATE_EXIT1;
    
    int in_env_EMMISION_RATE_EXIT2;
    
    int in_env_EMMISION_RATE_EXIT3;
    
    int in_env_EMMISION_RATE_EXIT4;
    
    int in_env_EMMISION_RATE_EXIT5;
    
    int in_env_EMMISION_RATE_EXIT6;
    
    int in_env_EMMISION_RATE_EXIT7;
    
    int in_env_EXIT1_PROBABILITY;
    
    int in_env_EXIT2_PROBABILITY;
    
    int in_env_EXIT3_PROBABILITY;
    
    int in_env_EXIT4_PROBABILITY;
    
    int in_env_EXIT5_PROBABILITY;
    
    int in_env_EXIT6_PROBABILITY;
    
    int in_env_EXIT7_PROBABILITY;
    
    int in_env_EXIT1_STATE;
    
    int in_env_EXIT2_STATE;
    
    int in_env_EXIT3_STATE;
    
    int in_env_EXIT4_STATE;
    
    int in_env_EXIT5_STATE;
    
    int in_env_EXIT6_STATE;
    
    int in_env_EXIT7_STATE;
    
    int in_env_EXIT1_CELL_COUNT;
    
    int in_env_EXIT2_CELL_COUNT;
    
    int in_env_EXIT3_CELL_COUNT;
    
    int in_env_EXIT4_CELL_COUNT;
    
    int in_env_EXIT5_CELL_COUNT;
    
    int in_env_EXIT6_CELL_COUNT;
    
    int in_env_EXIT7_CELL_COUNT;
    
    int in_env_TIME_SCALER;
    
    int in_env_STEER_WEIGHT;
    
    int in_env_AVOID_WEIGHT;
    
    int in_env_COLLISION_WEIGHT;
    
    int in_env_GOAL_WEIGHT;
    
	/* set agent count to zero */
	*h_xmachine_memory_agent_count = 0;
	*h_xmachine_memory_navmap_count = 0;
	
	/* Variables for initial state data */
	float agent_x;
	float agent_y;
	float agent_velx;
	float agent_vely;
	float agent_steer_x;
	float agent_steer_y;
	float agent_height;
	int agent_exit_no;
	float agent_speed;
	int agent_lod;
	float agent_animate;
	int agent_animate_dir;
	int navmap_x;
	int navmap_y;
	int navmap_exit_no;
	float navmap_height;
	float navmap_collision_x;
	float navmap_collision_y;
	float navmap_exit0_x;
	float navmap_exit0_y;
	float navmap_exit1_x;
	float navmap_exit1_y;
	float navmap_exit2_x;
	float navmap_exit2_y;
	float navmap_exit3_x;
	float navmap_exit3_y;
	float navmap_exit4_x;
	float navmap_exit4_y;
	float navmap_exit5_x;
	float navmap_exit5_y;
	float navmap_exit6_x;
	float navmap_exit6_y;

    /* Variables for environment variables */
    float env_EMMISION_RATE_EXIT1;
    float env_EMMISION_RATE_EXIT2;
    float env_EMMISION_RATE_EXIT3;
    float env_EMMISION_RATE_EXIT4;
    float env_EMMISION_RATE_EXIT5;
    float env_EMMISION_RATE_EXIT6;
    float env_EMMISION_RATE_EXIT7;
    int env_EXIT1_PROBABILITY;
    int env_EXIT2_PROBABILITY;
    int env_EXIT3_PROBABILITY;
    int env_EXIT4_PROBABILITY;
    int env_EXIT5_PROBABILITY;
    int env_EXIT6_PROBABILITY;
    int env_EXIT7_PROBABILITY;
    int env_EXIT1_STATE;
    int env_EXIT2_STATE;
    int env_EXIT3_STATE;
    int env_EXIT4_STATE;
    int env_EXIT5_STATE;
    int env_EXIT6_STATE;
    int env_EXIT7_STATE;
    int env_EXIT1_CELL_COUNT;
    int env_EXIT2_CELL_COUNT;
    int env_EXIT3_CELL_COUNT;
    int env_EXIT4_CELL_COUNT;
    int env_EXIT5_CELL_COUNT;
    int env_EXIT6_CELL_COUNT;
    int env_EXIT7_CELL_COUNT;
    float env_TIME_SCALER;
    float env_STEER_WEIGHT;
    float env_AVOID_WEIGHT;
    float env_COLLISION_WEIGHT;
    float env_GOAL_WEIGHT;
    


	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
    in_comment = 0;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;
	in_agent_x = 0;
	in_agent_y = 0;
	in_agent_velx = 0;
	in_agent_vely = 0;
	in_agent_steer_x = 0;
	in_agent_steer_y = 0;
	in_agent_height = 0;
	in_agent_exit_no = 0;
	in_agent_speed = 0;
	in_agent_lod = 0;
	in_agent_animate = 0;
	in_agent_animate_dir = 0;
	in_navmap_x = 0;
	in_navmap_y = 0;
	in_navmap_exit_no = 0;
	in_navmap_height = 0;
	in_navmap_collision_x = 0;
	in_navmap_collision_y = 0;
	in_navmap_exit0_x = 0;
	in_navmap_exit0_y = 0;
	in_navmap_exit1_x = 0;
	in_navmap_exit1_y = 0;
	in_navmap_exit2_x = 0;
	in_navmap_exit2_y = 0;
	in_navmap_exit3_x = 0;
	in_navmap_exit3_y = 0;
	in_navmap_exit4_x = 0;
	in_navmap_exit4_y = 0;
	in_navmap_exit5_x = 0;
	in_navmap_exit5_y = 0;
	in_navmap_exit6_x = 0;
	in_navmap_exit6_y = 0;
    in_env_EMMISION_RATE_EXIT1 = 0;
    in_env_EMMISION_RATE_EXIT2 = 0;
    in_env_EMMISION_RATE_EXIT3 = 0;
    in_env_EMMISION_RATE_EXIT4 = 0;
    in_env_EMMISION_RATE_EXIT5 = 0;
    in_env_EMMISION_RATE_EXIT6 = 0;
    in_env_EMMISION_RATE_EXIT7 = 0;
    in_env_EXIT1_PROBABILITY = 0;
    in_env_EXIT2_PROBABILITY = 0;
    in_env_EXIT3_PROBABILITY = 0;
    in_env_EXIT4_PROBABILITY = 0;
    in_env_EXIT5_PROBABILITY = 0;
    in_env_EXIT6_PROBABILITY = 0;
    in_env_EXIT7_PROBABILITY = 0;
    in_env_EXIT1_STATE = 0;
    in_env_EXIT2_STATE = 0;
    in_env_EXIT3_STATE = 0;
    in_env_EXIT4_STATE = 0;
    in_env_EXIT5_STATE = 0;
    in_env_EXIT6_STATE = 0;
    in_env_EXIT7_STATE = 0;
    in_env_EXIT1_CELL_COUNT = 0;
    in_env_EXIT2_CELL_COUNT = 0;
    in_env_EXIT3_CELL_COUNT = 0;
    in_env_EXIT4_CELL_COUNT = 0;
    in_env_EXIT5_CELL_COUNT = 0;
    in_env_EXIT6_CELL_COUNT = 0;
    in_env_EXIT7_CELL_COUNT = 0;
    in_env_TIME_SCALER = 0;
    in_env_STEER_WEIGHT = 0;
    in_env_AVOID_WEIGHT = 0;
    in_env_COLLISION_WEIGHT = 0;
    in_env_GOAL_WEIGHT = 0;
	//set all agent values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_agent_MAX; k++)
	{	
		h_agents->x[k] = 0;
		h_agents->y[k] = 0;
		h_agents->velx[k] = 0;
		h_agents->vely[k] = 0;
		h_agents->steer_x[k] = 0;
		h_agents->steer_y[k] = 0;
		h_agents->height[k] = 0;
		h_agents->exit_no[k] = 0;
		h_agents->speed[k] = 0;
		h_agents->lod[k] = 0;
		h_agents->animate[k] = 0;
		h_agents->animate_dir[k] = 0;
	}
	
	//set all navmap values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_navmap_MAX; k++)
	{	
		h_navmaps->x[k] = 0;
		h_navmaps->y[k] = 0;
		h_navmaps->exit_no[k] = 0;
		h_navmaps->height[k] = 0;
		h_navmaps->collision_x[k] = 0;
		h_navmaps->collision_y[k] = 0;
		h_navmaps->exit0_x[k] = 0;
		h_navmaps->exit0_y[k] = 0;
		h_navmaps->exit1_x[k] = 0;
		h_navmaps->exit1_y[k] = 0;
		h_navmaps->exit2_x[k] = 0;
		h_navmaps->exit2_y[k] = 0;
		h_navmaps->exit3_x[k] = 0;
		h_navmaps->exit3_y[k] = 0;
		h_navmaps->exit4_x[k] = 0;
		h_navmaps->exit4_y[k] = 0;
		h_navmaps->exit5_x[k] = 0;
		h_navmaps->exit5_y[k] = 0;
		h_navmaps->exit6_x[k] = 0;
		h_navmaps->exit6_y[k] = 0;
	}
	

	/* Default variables for memory */
    agent_x = 0;
    agent_y = 0;
    agent_velx = 0;
    agent_vely = 0;
    agent_steer_x = 0;
    agent_steer_y = 0;
    agent_height = 0;
    agent_exit_no = 0;
    agent_speed = 0;
    agent_lod = 0;
    agent_animate = 0;
    agent_animate_dir = 0;
    navmap_x = 0;
    navmap_y = 0;
    navmap_exit_no = 0;
    navmap_height = 0;
    navmap_collision_x = 0;
    navmap_collision_y = 0;
    navmap_exit0_x = 0;
    navmap_exit0_y = 0;
    navmap_exit1_x = 0;
    navmap_exit1_y = 0;
    navmap_exit2_x = 0;
    navmap_exit2_y = 0;
    navmap_exit3_x = 0;
    navmap_exit3_y = 0;
    navmap_exit4_x = 0;
    navmap_exit4_y = 0;
    navmap_exit5_x = 0;
    navmap_exit5_y = 0;
    navmap_exit6_x = 0;
    navmap_exit6_y = 0;

    /* Default variables for environment variables */
    env_EMMISION_RATE_EXIT1 = 0;
    env_EMMISION_RATE_EXIT2 = 0;
    env_EMMISION_RATE_EXIT3 = 0;
    env_EMMISION_RATE_EXIT4 = 0;
    env_EMMISION_RATE_EXIT5 = 0;
    env_EMMISION_RATE_EXIT6 = 0;
    env_EMMISION_RATE_EXIT7 = 0;
    env_EXIT1_PROBABILITY = 0;
    env_EXIT2_PROBABILITY = 0;
    env_EXIT3_PROBABILITY = 0;
    env_EXIT4_PROBABILITY = 0;
    env_EXIT5_PROBABILITY = 0;
    env_EXIT6_PROBABILITY = 0;
    env_EXIT7_PROBABILITY = 0;
    env_EXIT1_STATE = 0;
    env_EXIT2_STATE = 0;
    env_EXIT3_STATE = 0;
    env_EXIT4_STATE = 0;
    env_EXIT5_STATE = 0;
    env_EXIT6_STATE = 0;
    env_EXIT7_STATE = 0;
    env_EXIT1_CELL_COUNT = 0;
    env_EXIT2_CELL_COUNT = 0;
    env_EXIT3_CELL_COUNT = 0;
    env_EXIT4_CELL_COUNT = 0;
    env_EXIT5_CELL_COUNT = 0;
    env_EXIT6_CELL_COUNT = 0;
    env_EXIT7_CELL_COUNT = 0;
    env_TIME_SCALER = 0;
    env_STEER_WEIGHT = 0;
    env_AVOID_WEIGHT = 0;
    env_COLLISION_WEIGHT = 0;
    env_GOAL_WEIGHT = 0;
    
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            exit(EXIT_FAILURE);
        }

		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a > check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '>' && i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '>' && i >= 2 && buffer[i-1] == '-' && buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
		/* If the end of a tag */
		else if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "agent") == 0)
				{
					if (*h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent agent exceeded whilst reading data\n", xmachine_memory_agent_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_agents->x[*h_xmachine_memory_agent_count] = agent_x;//Check maximum x value
                    if(agent_maximum.x < agent_x)
                        agent_maximum.x = (float)agent_x;
                    //Check minimum x value
                    if(agent_minimum.x > agent_x)
                        agent_minimum.x = (float)agent_x;
                    
					h_agents->y[*h_xmachine_memory_agent_count] = agent_y;//Check maximum y value
                    if(agent_maximum.y < agent_y)
                        agent_maximum.y = (float)agent_y;
                    //Check minimum y value
                    if(agent_minimum.y > agent_y)
                        agent_minimum.y = (float)agent_y;
                    
					h_agents->velx[*h_xmachine_memory_agent_count] = agent_velx;
					h_agents->vely[*h_xmachine_memory_agent_count] = agent_vely;
					h_agents->steer_x[*h_xmachine_memory_agent_count] = agent_steer_x;
					h_agents->steer_y[*h_xmachine_memory_agent_count] = agent_steer_y;
					h_agents->height[*h_xmachine_memory_agent_count] = agent_height;
					h_agents->exit_no[*h_xmachine_memory_agent_count] = agent_exit_no;
					h_agents->speed[*h_xmachine_memory_agent_count] = agent_speed;
					h_agents->lod[*h_xmachine_memory_agent_count] = agent_lod;
					h_agents->animate[*h_xmachine_memory_agent_count] = agent_animate;
					h_agents->animate_dir[*h_xmachine_memory_agent_count] = agent_animate_dir;
					(*h_xmachine_memory_agent_count) ++;	
				}
				else if(strcmp(agentname, "navmap") == 0)
				{
					if (*h_xmachine_memory_navmap_count > xmachine_memory_navmap_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent navmap exceeded whilst reading data\n", xmachine_memory_navmap_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_navmaps->x[*h_xmachine_memory_navmap_count] = navmap_x;//Check maximum x value
                    if(agent_maximum.x < navmap_x)
                        agent_maximum.x = (float)navmap_x;
                    //Check minimum x value
                    if(agent_minimum.x > navmap_x)
                        agent_minimum.x = (float)navmap_x;
                    
					h_navmaps->y[*h_xmachine_memory_navmap_count] = navmap_y;//Check maximum y value
                    if(agent_maximum.y < navmap_y)
                        agent_maximum.y = (float)navmap_y;
                    //Check minimum y value
                    if(agent_minimum.y > navmap_y)
                        agent_minimum.y = (float)navmap_y;
                    
					h_navmaps->exit_no[*h_xmachine_memory_navmap_count] = navmap_exit_no;
					h_navmaps->height[*h_xmachine_memory_navmap_count] = navmap_height;
					h_navmaps->collision_x[*h_xmachine_memory_navmap_count] = navmap_collision_x;
					h_navmaps->collision_y[*h_xmachine_memory_navmap_count] = navmap_collision_y;
					h_navmaps->exit0_x[*h_xmachine_memory_navmap_count] = navmap_exit0_x;
					h_navmaps->exit0_y[*h_xmachine_memory_navmap_count] = navmap_exit0_y;
					h_navmaps->exit1_x[*h_xmachine_memory_navmap_count] = navmap_exit1_x;
					h_navmaps->exit1_y[*h_xmachine_memory_navmap_count] = navmap_exit1_y;
					h_navmaps->exit2_x[*h_xmachine_memory_navmap_count] = navmap_exit2_x;
					h_navmaps->exit2_y[*h_xmachine_memory_navmap_count] = navmap_exit2_y;
					h_navmaps->exit3_x[*h_xmachine_memory_navmap_count] = navmap_exit3_x;
					h_navmaps->exit3_y[*h_xmachine_memory_navmap_count] = navmap_exit3_y;
					h_navmaps->exit4_x[*h_xmachine_memory_navmap_count] = navmap_exit4_x;
					h_navmaps->exit4_y[*h_xmachine_memory_navmap_count] = navmap_exit4_y;
					h_navmaps->exit5_x[*h_xmachine_memory_navmap_count] = navmap_exit5_x;
					h_navmaps->exit5_y[*h_xmachine_memory_navmap_count] = navmap_exit5_y;
					h_navmaps->exit6_x[*h_xmachine_memory_navmap_count] = navmap_exit6_x;
					h_navmaps->exit6_y[*h_xmachine_memory_navmap_count] = navmap_exit6_y;
					(*h_xmachine_memory_navmap_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                agent_x = 0;
                agent_y = 0;
                agent_velx = 0;
                agent_vely = 0;
                agent_steer_x = 0;
                agent_steer_y = 0;
                agent_height = 0;
                agent_exit_no = 0;
                agent_speed = 0;
                agent_lod = 0;
                agent_animate = 0;
                agent_animate_dir = 0;
                navmap_x = 0;
                navmap_y = 0;
                navmap_exit_no = 0;
                navmap_height = 0;
                navmap_collision_x = 0;
                navmap_collision_y = 0;
                navmap_exit0_x = 0;
                navmap_exit0_y = 0;
                navmap_exit1_x = 0;
                navmap_exit1_y = 0;
                navmap_exit2_x = 0;
                navmap_exit2_y = 0;
                navmap_exit3_x = 0;
                navmap_exit3_y = 0;
                navmap_exit4_x = 0;
                navmap_exit4_y = 0;
                navmap_exit5_x = 0;
                navmap_exit5_y = 0;
                navmap_exit6_x = 0;
                navmap_exit6_y = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "x") == 0) in_agent_x = 1;
			if(strcmp(buffer, "/x") == 0) in_agent_x = 0;
			if(strcmp(buffer, "y") == 0) in_agent_y = 1;
			if(strcmp(buffer, "/y") == 0) in_agent_y = 0;
			if(strcmp(buffer, "velx") == 0) in_agent_velx = 1;
			if(strcmp(buffer, "/velx") == 0) in_agent_velx = 0;
			if(strcmp(buffer, "vely") == 0) in_agent_vely = 1;
			if(strcmp(buffer, "/vely") == 0) in_agent_vely = 0;
			if(strcmp(buffer, "steer_x") == 0) in_agent_steer_x = 1;
			if(strcmp(buffer, "/steer_x") == 0) in_agent_steer_x = 0;
			if(strcmp(buffer, "steer_y") == 0) in_agent_steer_y = 1;
			if(strcmp(buffer, "/steer_y") == 0) in_agent_steer_y = 0;
			if(strcmp(buffer, "height") == 0) in_agent_height = 1;
			if(strcmp(buffer, "/height") == 0) in_agent_height = 0;
			if(strcmp(buffer, "exit_no") == 0) in_agent_exit_no = 1;
			if(strcmp(buffer, "/exit_no") == 0) in_agent_exit_no = 0;
			if(strcmp(buffer, "speed") == 0) in_agent_speed = 1;
			if(strcmp(buffer, "/speed") == 0) in_agent_speed = 0;
			if(strcmp(buffer, "lod") == 0) in_agent_lod = 1;
			if(strcmp(buffer, "/lod") == 0) in_agent_lod = 0;
			if(strcmp(buffer, "animate") == 0) in_agent_animate = 1;
			if(strcmp(buffer, "/animate") == 0) in_agent_animate = 0;
			if(strcmp(buffer, "animate_dir") == 0) in_agent_animate_dir = 1;
			if(strcmp(buffer, "/animate_dir") == 0) in_agent_animate_dir = 0;
			if(strcmp(buffer, "x") == 0) in_navmap_x = 1;
			if(strcmp(buffer, "/x") == 0) in_navmap_x = 0;
			if(strcmp(buffer, "y") == 0) in_navmap_y = 1;
			if(strcmp(buffer, "/y") == 0) in_navmap_y = 0;
			if(strcmp(buffer, "exit_no") == 0) in_navmap_exit_no = 1;
			if(strcmp(buffer, "/exit_no") == 0) in_navmap_exit_no = 0;
			if(strcmp(buffer, "height") == 0) in_navmap_height = 1;
			if(strcmp(buffer, "/height") == 0) in_navmap_height = 0;
			if(strcmp(buffer, "collision_x") == 0) in_navmap_collision_x = 1;
			if(strcmp(buffer, "/collision_x") == 0) in_navmap_collision_x = 0;
			if(strcmp(buffer, "collision_y") == 0) in_navmap_collision_y = 1;
			if(strcmp(buffer, "/collision_y") == 0) in_navmap_collision_y = 0;
			if(strcmp(buffer, "exit0_x") == 0) in_navmap_exit0_x = 1;
			if(strcmp(buffer, "/exit0_x") == 0) in_navmap_exit0_x = 0;
			if(strcmp(buffer, "exit0_y") == 0) in_navmap_exit0_y = 1;
			if(strcmp(buffer, "/exit0_y") == 0) in_navmap_exit0_y = 0;
			if(strcmp(buffer, "exit1_x") == 0) in_navmap_exit1_x = 1;
			if(strcmp(buffer, "/exit1_x") == 0) in_navmap_exit1_x = 0;
			if(strcmp(buffer, "exit1_y") == 0) in_navmap_exit1_y = 1;
			if(strcmp(buffer, "/exit1_y") == 0) in_navmap_exit1_y = 0;
			if(strcmp(buffer, "exit2_x") == 0) in_navmap_exit2_x = 1;
			if(strcmp(buffer, "/exit2_x") == 0) in_navmap_exit2_x = 0;
			if(strcmp(buffer, "exit2_y") == 0) in_navmap_exit2_y = 1;
			if(strcmp(buffer, "/exit2_y") == 0) in_navmap_exit2_y = 0;
			if(strcmp(buffer, "exit3_x") == 0) in_navmap_exit3_x = 1;
			if(strcmp(buffer, "/exit3_x") == 0) in_navmap_exit3_x = 0;
			if(strcmp(buffer, "exit3_y") == 0) in_navmap_exit3_y = 1;
			if(strcmp(buffer, "/exit3_y") == 0) in_navmap_exit3_y = 0;
			if(strcmp(buffer, "exit4_x") == 0) in_navmap_exit4_x = 1;
			if(strcmp(buffer, "/exit4_x") == 0) in_navmap_exit4_x = 0;
			if(strcmp(buffer, "exit4_y") == 0) in_navmap_exit4_y = 1;
			if(strcmp(buffer, "/exit4_y") == 0) in_navmap_exit4_y = 0;
			if(strcmp(buffer, "exit5_x") == 0) in_navmap_exit5_x = 1;
			if(strcmp(buffer, "/exit5_x") == 0) in_navmap_exit5_x = 0;
			if(strcmp(buffer, "exit5_y") == 0) in_navmap_exit5_y = 1;
			if(strcmp(buffer, "/exit5_y") == 0) in_navmap_exit5_y = 0;
			if(strcmp(buffer, "exit6_x") == 0) in_navmap_exit6_x = 1;
			if(strcmp(buffer, "/exit6_x") == 0) in_navmap_exit6_x = 0;
			if(strcmp(buffer, "exit6_y") == 0) in_navmap_exit6_y = 1;
			if(strcmp(buffer, "/exit6_y") == 0) in_navmap_exit6_y = 0;
			
            /* environment variables */
            if(strcmp(buffer, "EMMISION_RATE_EXIT1") == 0) in_env_EMMISION_RATE_EXIT1 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT1") == 0) in_env_EMMISION_RATE_EXIT1 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT2") == 0) in_env_EMMISION_RATE_EXIT2 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT2") == 0) in_env_EMMISION_RATE_EXIT2 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT3") == 0) in_env_EMMISION_RATE_EXIT3 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT3") == 0) in_env_EMMISION_RATE_EXIT3 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT4") == 0) in_env_EMMISION_RATE_EXIT4 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT4") == 0) in_env_EMMISION_RATE_EXIT4 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT5") == 0) in_env_EMMISION_RATE_EXIT5 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT5") == 0) in_env_EMMISION_RATE_EXIT5 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT6") == 0) in_env_EMMISION_RATE_EXIT6 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT6") == 0) in_env_EMMISION_RATE_EXIT6 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT7") == 0) in_env_EMMISION_RATE_EXIT7 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT7") == 0) in_env_EMMISION_RATE_EXIT7 = 0;
			if(strcmp(buffer, "EXIT1_PROBABILITY") == 0) in_env_EXIT1_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT1_PROBABILITY") == 0) in_env_EXIT1_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT2_PROBABILITY") == 0) in_env_EXIT2_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT2_PROBABILITY") == 0) in_env_EXIT2_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT3_PROBABILITY") == 0) in_env_EXIT3_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT3_PROBABILITY") == 0) in_env_EXIT3_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT4_PROBABILITY") == 0) in_env_EXIT4_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT4_PROBABILITY") == 0) in_env_EXIT4_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT5_PROBABILITY") == 0) in_env_EXIT5_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT5_PROBABILITY") == 0) in_env_EXIT5_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT6_PROBABILITY") == 0) in_env_EXIT6_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT6_PROBABILITY") == 0) in_env_EXIT6_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT7_PROBABILITY") == 0) in_env_EXIT7_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT7_PROBABILITY") == 0) in_env_EXIT7_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT1_STATE") == 0) in_env_EXIT1_STATE = 1;
            if(strcmp(buffer, "/EXIT1_STATE") == 0) in_env_EXIT1_STATE = 0;
			if(strcmp(buffer, "EXIT2_STATE") == 0) in_env_EXIT2_STATE = 1;
            if(strcmp(buffer, "/EXIT2_STATE") == 0) in_env_EXIT2_STATE = 0;
			if(strcmp(buffer, "EXIT3_STATE") == 0) in_env_EXIT3_STATE = 1;
            if(strcmp(buffer, "/EXIT3_STATE") == 0) in_env_EXIT3_STATE = 0;
			if(strcmp(buffer, "EXIT4_STATE") == 0) in_env_EXIT4_STATE = 1;
            if(strcmp(buffer, "/EXIT4_STATE") == 0) in_env_EXIT4_STATE = 0;
			if(strcmp(buffer, "EXIT5_STATE") == 0) in_env_EXIT5_STATE = 1;
            if(strcmp(buffer, "/EXIT5_STATE") == 0) in_env_EXIT5_STATE = 0;
			if(strcmp(buffer, "EXIT6_STATE") == 0) in_env_EXIT6_STATE = 1;
            if(strcmp(buffer, "/EXIT6_STATE") == 0) in_env_EXIT6_STATE = 0;
			if(strcmp(buffer, "EXIT7_STATE") == 0) in_env_EXIT7_STATE = 1;
            if(strcmp(buffer, "/EXIT7_STATE") == 0) in_env_EXIT7_STATE = 0;
			if(strcmp(buffer, "EXIT1_CELL_COUNT") == 0) in_env_EXIT1_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT1_CELL_COUNT") == 0) in_env_EXIT1_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT2_CELL_COUNT") == 0) in_env_EXIT2_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT2_CELL_COUNT") == 0) in_env_EXIT2_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT3_CELL_COUNT") == 0) in_env_EXIT3_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT3_CELL_COUNT") == 0) in_env_EXIT3_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT4_CELL_COUNT") == 0) in_env_EXIT4_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT4_CELL_COUNT") == 0) in_env_EXIT4_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT5_CELL_COUNT") == 0) in_env_EXIT5_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT5_CELL_COUNT") == 0) in_env_EXIT5_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT6_CELL_COUNT") == 0) in_env_EXIT6_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT6_CELL_COUNT") == 0) in_env_EXIT6_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT7_CELL_COUNT") == 0) in_env_EXIT7_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT7_CELL_COUNT") == 0) in_env_EXIT7_CELL_COUNT = 0;
			if(strcmp(buffer, "TIME_SCALER") == 0) in_env_TIME_SCALER = 1;
            if(strcmp(buffer, "/TIME_SCALER") == 0) in_env_TIME_SCALER = 0;
			if(strcmp(buffer, "STEER_WEIGHT") == 0) in_env_STEER_WEIGHT = 1;
            if(strcmp(buffer, "/STEER_WEIGHT") == 0) in_env_STEER_WEIGHT = 0;
			if(strcmp(buffer, "AVOID_WEIGHT") == 0) in_env_AVOID_WEIGHT = 1;
            if(strcmp(buffer, "/AVOID_WEIGHT") == 0) in_env_AVOID_WEIGHT = 0;
			if(strcmp(buffer, "COLLISION_WEIGHT") == 0) in_env_COLLISION_WEIGHT = 1;
            if(strcmp(buffer, "/COLLISION_WEIGHT") == 0) in_env_COLLISION_WEIGHT = 0;
			if(strcmp(buffer, "GOAL_WEIGHT") == 0) in_env_GOAL_WEIGHT = 1;
            if(strcmp(buffer, "/GOAL_WEIGHT") == 0) in_env_GOAL_WEIGHT = 0;
			

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else if (in_xagent)
			{
				if(in_agent_x){
                    agent_x = (float) fgpu_atof(buffer); 
                }
				if(in_agent_y){
                    agent_y = (float) fgpu_atof(buffer); 
                }
				if(in_agent_velx){
                    agent_velx = (float) fgpu_atof(buffer); 
                }
				if(in_agent_vely){
                    agent_vely = (float) fgpu_atof(buffer); 
                }
				if(in_agent_steer_x){
                    agent_steer_x = (float) fgpu_atof(buffer); 
                }
				if(in_agent_steer_y){
                    agent_steer_y = (float) fgpu_atof(buffer); 
                }
				if(in_agent_height){
                    agent_height = (float) fgpu_atof(buffer); 
                }
				if(in_agent_exit_no){
                    agent_exit_no = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_speed){
                    agent_speed = (float) fgpu_atof(buffer); 
                }
				if(in_agent_lod){
                    agent_lod = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_animate){
                    agent_animate = (float) fgpu_atof(buffer); 
                }
				if(in_agent_animate_dir){
                    agent_animate_dir = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_x){
                    navmap_x = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_y){
                    navmap_y = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_exit_no){
                    navmap_exit_no = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_height){
                    navmap_height = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_collision_x){
                    navmap_collision_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_collision_y){
                    navmap_collision_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit0_x){
                    navmap_exit0_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit0_y){
                    navmap_exit0_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit1_x){
                    navmap_exit1_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit1_y){
                    navmap_exit1_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit2_x){
                    navmap_exit2_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit2_y){
                    navmap_exit2_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit3_x){
                    navmap_exit3_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit3_y){
                    navmap_exit3_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit4_x){
                    navmap_exit4_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit4_y){
                    navmap_exit4_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit5_x){
                    navmap_exit5_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit5_y){
                    navmap_exit5_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit6_x){
                    navmap_exit6_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit6_y){
                    navmap_exit6_y = (float) fgpu_atof(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_EMMISION_RATE_EXIT1){
              
                    env_EMMISION_RATE_EXIT1 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT1(&env_EMMISION_RATE_EXIT1);
                  
              }
            if(in_env_EMMISION_RATE_EXIT2){
              
                    env_EMMISION_RATE_EXIT2 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT2(&env_EMMISION_RATE_EXIT2);
                  
              }
            if(in_env_EMMISION_RATE_EXIT3){
              
                    env_EMMISION_RATE_EXIT3 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT3(&env_EMMISION_RATE_EXIT3);
                  
              }
            if(in_env_EMMISION_RATE_EXIT4){
              
                    env_EMMISION_RATE_EXIT4 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT4(&env_EMMISION_RATE_EXIT4);
                  
              }
            if(in_env_EMMISION_RATE_EXIT5){
              
                    env_EMMISION_RATE_EXIT5 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT5(&env_EMMISION_RATE_EXIT5);
                  
              }
            if(in_env_EMMISION_RATE_EXIT6){
              
                    env_EMMISION_RATE_EXIT6 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT6(&env_EMMISION_RATE_EXIT6);
                  
              }
            if(in_env_EMMISION_RATE_EXIT7){
              
                    env_EMMISION_RATE_EXIT7 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT7(&env_EMMISION_RATE_EXIT7);
                  
              }
            if(in_env_EXIT1_PROBABILITY){
              
                    env_EXIT1_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_PROBABILITY(&env_EXIT1_PROBABILITY);
                  
              }
            if(in_env_EXIT2_PROBABILITY){
              
                    env_EXIT2_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_PROBABILITY(&env_EXIT2_PROBABILITY);
                  
              }
            if(in_env_EXIT3_PROBABILITY){
              
                    env_EXIT3_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_PROBABILITY(&env_EXIT3_PROBABILITY);
                  
              }
            if(in_env_EXIT4_PROBABILITY){
              
                    env_EXIT4_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_PROBABILITY(&env_EXIT4_PROBABILITY);
                  
              }
            if(in_env_EXIT5_PROBABILITY){
              
                    env_EXIT5_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_PROBABILITY(&env_EXIT5_PROBABILITY);
                  
              }
            if(in_env_EXIT6_PROBABILITY){
              
                    env_EXIT6_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_PROBABILITY(&env_EXIT6_PROBABILITY);
                  
              }
            if(in_env_EXIT7_PROBABILITY){
              
                    env_EXIT7_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_PROBABILITY(&env_EXIT7_PROBABILITY);
                  
              }
            if(in_env_EXIT1_STATE){
              
                    env_EXIT1_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_STATE(&env_EXIT1_STATE);
                  
              }
            if(in_env_EXIT2_STATE){
              
                    env_EXIT2_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_STATE(&env_EXIT2_STATE);
                  
              }
            if(in_env_EXIT3_STATE){
              
                    env_EXIT3_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_STATE(&env_EXIT3_STATE);
                  
              }
            if(in_env_EXIT4_STATE){
              
                    env_EXIT4_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_STATE(&env_EXIT4_STATE);
                  
              }
            if(in_env_EXIT5_STATE){
              
                    env_EXIT5_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_STATE(&env_EXIT5_STATE);
                  
              }
            if(in_env_EXIT6_STATE){
              
                    env_EXIT6_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_STATE(&env_EXIT6_STATE);
                  
              }
            if(in_env_EXIT7_STATE){
              
                    env_EXIT7_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_STATE(&env_EXIT7_STATE);
                  
              }
            if(in_env_EXIT1_CELL_COUNT){
              
                    env_EXIT1_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_CELL_COUNT(&env_EXIT1_CELL_COUNT);
                  
              }
            if(in_env_EXIT2_CELL_COUNT){
              
                    env_EXIT2_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_CELL_COUNT(&env_EXIT2_CELL_COUNT);
                  
              }
            if(in_env_EXIT3_CELL_COUNT){
              
                    env_EXIT3_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_CELL_COUNT(&env_EXIT3_CELL_COUNT);
                  
              }
            if(in_env_EXIT4_CELL_COUNT){
              
                    env_EXIT4_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_CELL_COUNT(&env_EXIT4_CELL_COUNT);
                  
              }
            if(in_env_EXIT5_CELL_COUNT){
              
                    env_EXIT5_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_CELL_COUNT(&env_EXIT5_CELL_COUNT);
                  
              }
            if(in_env_EXIT6_CELL_COUNT){
              
                    env_EXIT6_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_CELL_COUNT(&env_EXIT6_CELL_COUNT);
                  
              }
            if(in_env_EXIT7_CELL_COUNT){
              
                    env_EXIT7_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_CELL_COUNT(&env_EXIT7_CELL_COUNT);
                  
              }
            if(in_env_TIME_SCALER){
              
                    env_TIME_SCALER = (float) fgpu_atof(buffer);
                    
                    set_TIME_SCALER(&env_TIME_SCALER);
                  
              }
            if(in_env_STEER_WEIGHT){
              
                    env_STEER_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_STEER_WEIGHT(&env_STEER_WEIGHT);
                  
              }
            if(in_env_AVOID_WEIGHT){
              
                    env_AVOID_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_AVOID_WEIGHT(&env_AVOID_WEIGHT);
                  
              }
            if(in_env_COLLISION_WEIGHT){
              
                    env_COLLISION_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_COLLISION_WEIGHT(&env_COLLISION_WEIGHT);
                  
              }
            if(in_env_GOAL_WEIGHT){
              
                    env_GOAL_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_GOAL_WEIGHT(&env_GOAL_WEIGHT);
                  
              }
            
            }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 && c == '-' && buffer[1] == '-' && buffer[0] == '!'){
                in_comment = 1;
                // Reset the buffer and i.
                buffer[0] = 0;
                i = 0;
            }

            // Store the character and increment the counter
            buffer[i] = c;
            i++;

		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
        fflush(stdout);
    }

    // If the in_comment flag is still marked, issue a warning.
    if(in_comment){
        fprintf(stdout, "Warning: Un-terminated comment in %s\n", inputpath);
        fflush(stdout);
    }    

	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}


/* Methods to load static networks from disk */
