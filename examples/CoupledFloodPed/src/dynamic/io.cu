
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_FloodCell_list* h_FloodCells_Default, xmachine_memory_FloodCell_list* d_FloodCells_Default, int h_xmachine_memory_FloodCell_Default_count,xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_FloodCells_Default, d_FloodCells_Default, sizeof(xmachine_memory_FloodCell_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying FloodCell Agent Default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
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
    
    fputs("\t<xmin>", file);
    sprintf(data, "%f", (*get_xmin()));
    fputs(data, file);
    fputs("</xmin>\n", file);
    fputs("\t<xmax>", file);
    sprintf(data, "%f", (*get_xmax()));
    fputs(data, file);
    fputs("</xmax>\n", file);
    fputs("\t<ymin>", file);
    sprintf(data, "%f", (*get_ymin()));
    fputs(data, file);
    fputs("</ymin>\n", file);
    fputs("\t<ymax>", file);
    sprintf(data, "%f", (*get_ymax()));
    fputs(data, file);
    fputs("</ymax>\n", file);
    fputs("\t<dt>", file);
    sprintf(data, "%f", (*get_dt()));
    fputs(data, file);
    fputs("</dt>\n", file);
    fputs("\t<DXL>", file);
    sprintf(data, "%f", (*get_DXL()));
    fputs(data, file);
    fputs("</DXL>\n", file);
    fputs("\t<DYL>", file);
    sprintf(data, "%f", (*get_DYL()));
    fputs(data, file);
    fputs("</DYL>\n", file);
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

	//Write each FloodCell agent to xml
	for (int i=0; i<h_xmachine_memory_FloodCell_Default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>FloodCell</name>\n", file);
        
		fputs("<inDomain>", file);
        sprintf(data, "%d", h_FloodCells_Default->inDomain[i]);
		fputs(data, file);
		fputs("</inDomain>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_FloodCells_Default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_FloodCells_Default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z0>", file);
        sprintf(data, "%f", h_FloodCells_Default->z0[i]);
		fputs(data, file);
		fputs("</z0>\n", file);
        
		fputs("<h>", file);
        sprintf(data, "%f", h_FloodCells_Default->h[i]);
		fputs(data, file);
		fputs("</h>\n", file);
        
		fputs("<qx>", file);
        sprintf(data, "%f", h_FloodCells_Default->qx[i]);
		fputs(data, file);
		fputs("</qx>\n", file);
        
		fputs("<qy>", file);
        sprintf(data, "%f", h_FloodCells_Default->qy[i]);
		fputs(data, file);
		fputs("</qy>\n", file);
        
		fputs("<timeStep>", file);
        sprintf(data, "%f", h_FloodCells_Default->timeStep[i]);
		fputs(data, file);
		fputs("</timeStep>\n", file);
        
		fputs("<minh_loc>", file);
        sprintf(data, "%f", h_FloodCells_Default->minh_loc[i]);
		fputs(data, file);
		fputs("</minh_loc>\n", file);
        
		fputs("<hFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_E[i]);
		fputs(data, file);
		fputs("</hFace_E>\n", file);
        
		fputs("<etFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_E[i]);
		fputs(data, file);
		fputs("</etFace_E>\n", file);
        
		fputs("<qxFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_E[i]);
		fputs(data, file);
		fputs("</qxFace_E>\n", file);
        
		fputs("<qyFace_E>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_E[i]);
		fputs(data, file);
		fputs("</qyFace_E>\n", file);
        
		fputs("<hFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_W[i]);
		fputs(data, file);
		fputs("</hFace_W>\n", file);
        
		fputs("<etFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_W[i]);
		fputs(data, file);
		fputs("</etFace_W>\n", file);
        
		fputs("<qxFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_W[i]);
		fputs(data, file);
		fputs("</qxFace_W>\n", file);
        
		fputs("<qyFace_W>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_W[i]);
		fputs(data, file);
		fputs("</qyFace_W>\n", file);
        
		fputs("<hFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_N[i]);
		fputs(data, file);
		fputs("</hFace_N>\n", file);
        
		fputs("<etFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_N[i]);
		fputs(data, file);
		fputs("</etFace_N>\n", file);
        
		fputs("<qxFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_N[i]);
		fputs(data, file);
		fputs("</qxFace_N>\n", file);
        
		fputs("<qyFace_N>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_N[i]);
		fputs(data, file);
		fputs("</qyFace_N>\n", file);
        
		fputs("<hFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->hFace_S[i]);
		fputs(data, file);
		fputs("</hFace_S>\n", file);
        
		fputs("<etFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->etFace_S[i]);
		fputs(data, file);
		fputs("</etFace_S>\n", file);
        
		fputs("<qxFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->qxFace_S[i]);
		fputs(data, file);
		fputs("</qxFace_S>\n", file);
        
		fputs("<qyFace_S>", file);
        sprintf(data, "%f", h_FloodCells_Default->qyFace_S[i]);
		fputs(data, file);
		fputs("</qyFace_S>\n", file);
        
		fputs("</xagent>\n", file);
	}
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

void readInitialStates(char* inputpath, xmachine_memory_FloodCell_list* h_FloodCells, int* h_xmachine_memory_FloodCell_count,xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count)
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
    int in_FloodCell_inDomain;
    int in_FloodCell_x;
    int in_FloodCell_y;
    int in_FloodCell_z0;
    int in_FloodCell_h;
    int in_FloodCell_qx;
    int in_FloodCell_qy;
    int in_FloodCell_timeStep;
    int in_FloodCell_minh_loc;
    int in_FloodCell_hFace_E;
    int in_FloodCell_etFace_E;
    int in_FloodCell_qxFace_E;
    int in_FloodCell_qyFace_E;
    int in_FloodCell_hFace_W;
    int in_FloodCell_etFace_W;
    int in_FloodCell_qxFace_W;
    int in_FloodCell_qyFace_W;
    int in_FloodCell_hFace_N;
    int in_FloodCell_etFace_N;
    int in_FloodCell_qxFace_N;
    int in_FloodCell_qyFace_N;
    int in_FloodCell_hFace_S;
    int in_FloodCell_etFace_S;
    int in_FloodCell_qxFace_S;
    int in_FloodCell_qyFace_S;
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
    int in_env_xmin;
    
    int in_env_xmax;
    
    int in_env_ymin;
    
    int in_env_ymax;
    
    int in_env_dt;
    
    int in_env_DXL;
    
    int in_env_DYL;
    
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
	*h_xmachine_memory_FloodCell_count = 0;
	*h_xmachine_memory_agent_count = 0;
	*h_xmachine_memory_navmap_count = 0;
	
	/* Variables for initial state data */
	int FloodCell_inDomain;
	int FloodCell_x;
	int FloodCell_y;
	double FloodCell_z0;
	double FloodCell_h;
	double FloodCell_qx;
	double FloodCell_qy;
	double FloodCell_timeStep;
	double FloodCell_minh_loc;
	double FloodCell_hFace_E;
	double FloodCell_etFace_E;
	double FloodCell_qxFace_E;
	double FloodCell_qyFace_E;
	double FloodCell_hFace_W;
	double FloodCell_etFace_W;
	double FloodCell_qxFace_W;
	double FloodCell_qyFace_W;
	double FloodCell_hFace_N;
	double FloodCell_etFace_N;
	double FloodCell_qxFace_N;
	double FloodCell_qyFace_N;
	double FloodCell_hFace_S;
	double FloodCell_etFace_S;
	double FloodCell_qxFace_S;
	double FloodCell_qyFace_S;
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
    double env_xmin;
    double env_xmax;
    double env_ymin;
    double env_ymax;
    double env_dt;
    double env_DXL;
    double env_DYL;
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
	in_FloodCell_inDomain = 0;
	in_FloodCell_x = 0;
	in_FloodCell_y = 0;
	in_FloodCell_z0 = 0;
	in_FloodCell_h = 0;
	in_FloodCell_qx = 0;
	in_FloodCell_qy = 0;
	in_FloodCell_timeStep = 0;
	in_FloodCell_minh_loc = 0;
	in_FloodCell_hFace_E = 0;
	in_FloodCell_etFace_E = 0;
	in_FloodCell_qxFace_E = 0;
	in_FloodCell_qyFace_E = 0;
	in_FloodCell_hFace_W = 0;
	in_FloodCell_etFace_W = 0;
	in_FloodCell_qxFace_W = 0;
	in_FloodCell_qyFace_W = 0;
	in_FloodCell_hFace_N = 0;
	in_FloodCell_etFace_N = 0;
	in_FloodCell_qxFace_N = 0;
	in_FloodCell_qyFace_N = 0;
	in_FloodCell_hFace_S = 0;
	in_FloodCell_etFace_S = 0;
	in_FloodCell_qxFace_S = 0;
	in_FloodCell_qyFace_S = 0;
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
    in_env_xmin = 0;
    in_env_xmax = 0;
    in_env_ymin = 0;
    in_env_ymax = 0;
    in_env_dt = 0;
    in_env_DXL = 0;
    in_env_DYL = 0;
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
	//set all FloodCell values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_FloodCell_MAX; k++)
	{	
		h_FloodCells->inDomain[k] = 0;
		h_FloodCells->x[k] = 0;
		h_FloodCells->y[k] = 0;
		h_FloodCells->z0[k] = 0;
		h_FloodCells->h[k] = 0;
		h_FloodCells->qx[k] = 0;
		h_FloodCells->qy[k] = 0;
		h_FloodCells->timeStep[k] = 0;
		h_FloodCells->minh_loc[k] = 0;
		h_FloodCells->hFace_E[k] = 0;
		h_FloodCells->etFace_E[k] = 0;
		h_FloodCells->qxFace_E[k] = 0;
		h_FloodCells->qyFace_E[k] = 0;
		h_FloodCells->hFace_W[k] = 0;
		h_FloodCells->etFace_W[k] = 0;
		h_FloodCells->qxFace_W[k] = 0;
		h_FloodCells->qyFace_W[k] = 0;
		h_FloodCells->hFace_N[k] = 0;
		h_FloodCells->etFace_N[k] = 0;
		h_FloodCells->qxFace_N[k] = 0;
		h_FloodCells->qyFace_N[k] = 0;
		h_FloodCells->hFace_S[k] = 0;
		h_FloodCells->etFace_S[k] = 0;
		h_FloodCells->qxFace_S[k] = 0;
		h_FloodCells->qyFace_S[k] = 0;
	}
	
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
    FloodCell_inDomain = 0;
    FloodCell_x = 0;
    FloodCell_y = 0;
    FloodCell_z0 = 0;
    FloodCell_h = 0;
    FloodCell_qx = 0;
    FloodCell_qy = 0;
    FloodCell_timeStep = 0;
    FloodCell_minh_loc = 0;
    FloodCell_hFace_E = 0;
    FloodCell_etFace_E = 0;
    FloodCell_qxFace_E = 0;
    FloodCell_qyFace_E = 0;
    FloodCell_hFace_W = 0;
    FloodCell_etFace_W = 0;
    FloodCell_qxFace_W = 0;
    FloodCell_qyFace_W = 0;
    FloodCell_hFace_N = 0;
    FloodCell_etFace_N = 0;
    FloodCell_qxFace_N = 0;
    FloodCell_qyFace_N = 0;
    FloodCell_hFace_S = 0;
    FloodCell_etFace_S = 0;
    FloodCell_qxFace_S = 0;
    FloodCell_qyFace_S = 0;
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
    env_xmin = 0;
    env_xmax = 0;
    env_ymin = 0;
    env_ymax = 0;
    env_dt = 0;
    env_DXL = 0;
    env_DYL = 0;
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
				if(strcmp(agentname, "FloodCell") == 0)
				{
					if (*h_xmachine_memory_FloodCell_count > xmachine_memory_FloodCell_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent FloodCell exceeded whilst reading data\n", xmachine_memory_FloodCell_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_FloodCells->inDomain[*h_xmachine_memory_FloodCell_count] = FloodCell_inDomain;
					h_FloodCells->x[*h_xmachine_memory_FloodCell_count] = FloodCell_x;//Check maximum x value
                    if(agent_maximum.x < FloodCell_x)
                        agent_maximum.x = (float)FloodCell_x;
                    //Check minimum x value
                    if(agent_minimum.x > FloodCell_x)
                        agent_minimum.x = (float)FloodCell_x;
                    
					h_FloodCells->y[*h_xmachine_memory_FloodCell_count] = FloodCell_y;//Check maximum y value
                    if(agent_maximum.y < FloodCell_y)
                        agent_maximum.y = (float)FloodCell_y;
                    //Check minimum y value
                    if(agent_minimum.y > FloodCell_y)
                        agent_minimum.y = (float)FloodCell_y;
                    
					h_FloodCells->z0[*h_xmachine_memory_FloodCell_count] = FloodCell_z0;
					h_FloodCells->h[*h_xmachine_memory_FloodCell_count] = FloodCell_h;
					h_FloodCells->qx[*h_xmachine_memory_FloodCell_count] = FloodCell_qx;
					h_FloodCells->qy[*h_xmachine_memory_FloodCell_count] = FloodCell_qy;
					h_FloodCells->timeStep[*h_xmachine_memory_FloodCell_count] = FloodCell_timeStep;
					h_FloodCells->minh_loc[*h_xmachine_memory_FloodCell_count] = FloodCell_minh_loc;
					h_FloodCells->hFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_E;
					h_FloodCells->etFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_E;
					h_FloodCells->qxFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_E;
					h_FloodCells->qyFace_E[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_E;
					h_FloodCells->hFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_W;
					h_FloodCells->etFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_W;
					h_FloodCells->qxFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_W;
					h_FloodCells->qyFace_W[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_W;
					h_FloodCells->hFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_N;
					h_FloodCells->etFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_N;
					h_FloodCells->qxFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_N;
					h_FloodCells->qyFace_N[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_N;
					h_FloodCells->hFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_hFace_S;
					h_FloodCells->etFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_etFace_S;
					h_FloodCells->qxFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_qxFace_S;
					h_FloodCells->qyFace_S[*h_xmachine_memory_FloodCell_count] = FloodCell_qyFace_S;
					(*h_xmachine_memory_FloodCell_count) ++;	
				}
				else if(strcmp(agentname, "agent") == 0)
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
                FloodCell_inDomain = 0;
                FloodCell_x = 0;
                FloodCell_y = 0;
                FloodCell_z0 = 0;
                FloodCell_h = 0;
                FloodCell_qx = 0;
                FloodCell_qy = 0;
                FloodCell_timeStep = 0;
                FloodCell_minh_loc = 0;
                FloodCell_hFace_E = 0;
                FloodCell_etFace_E = 0;
                FloodCell_qxFace_E = 0;
                FloodCell_qyFace_E = 0;
                FloodCell_hFace_W = 0;
                FloodCell_etFace_W = 0;
                FloodCell_qxFace_W = 0;
                FloodCell_qyFace_W = 0;
                FloodCell_hFace_N = 0;
                FloodCell_etFace_N = 0;
                FloodCell_qxFace_N = 0;
                FloodCell_qyFace_N = 0;
                FloodCell_hFace_S = 0;
                FloodCell_etFace_S = 0;
                FloodCell_qxFace_S = 0;
                FloodCell_qyFace_S = 0;
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
			if(strcmp(buffer, "inDomain") == 0) in_FloodCell_inDomain = 1;
			if(strcmp(buffer, "/inDomain") == 0) in_FloodCell_inDomain = 0;
			if(strcmp(buffer, "x") == 0) in_FloodCell_x = 1;
			if(strcmp(buffer, "/x") == 0) in_FloodCell_x = 0;
			if(strcmp(buffer, "y") == 0) in_FloodCell_y = 1;
			if(strcmp(buffer, "/y") == 0) in_FloodCell_y = 0;
			if(strcmp(buffer, "z0") == 0) in_FloodCell_z0 = 1;
			if(strcmp(buffer, "/z0") == 0) in_FloodCell_z0 = 0;
			if(strcmp(buffer, "h") == 0) in_FloodCell_h = 1;
			if(strcmp(buffer, "/h") == 0) in_FloodCell_h = 0;
			if(strcmp(buffer, "qx") == 0) in_FloodCell_qx = 1;
			if(strcmp(buffer, "/qx") == 0) in_FloodCell_qx = 0;
			if(strcmp(buffer, "qy") == 0) in_FloodCell_qy = 1;
			if(strcmp(buffer, "/qy") == 0) in_FloodCell_qy = 0;
			if(strcmp(buffer, "timeStep") == 0) in_FloodCell_timeStep = 1;
			if(strcmp(buffer, "/timeStep") == 0) in_FloodCell_timeStep = 0;
			if(strcmp(buffer, "minh_loc") == 0) in_FloodCell_minh_loc = 1;
			if(strcmp(buffer, "/minh_loc") == 0) in_FloodCell_minh_loc = 0;
			if(strcmp(buffer, "hFace_E") == 0) in_FloodCell_hFace_E = 1;
			if(strcmp(buffer, "/hFace_E") == 0) in_FloodCell_hFace_E = 0;
			if(strcmp(buffer, "etFace_E") == 0) in_FloodCell_etFace_E = 1;
			if(strcmp(buffer, "/etFace_E") == 0) in_FloodCell_etFace_E = 0;
			if(strcmp(buffer, "qxFace_E") == 0) in_FloodCell_qxFace_E = 1;
			if(strcmp(buffer, "/qxFace_E") == 0) in_FloodCell_qxFace_E = 0;
			if(strcmp(buffer, "qyFace_E") == 0) in_FloodCell_qyFace_E = 1;
			if(strcmp(buffer, "/qyFace_E") == 0) in_FloodCell_qyFace_E = 0;
			if(strcmp(buffer, "hFace_W") == 0) in_FloodCell_hFace_W = 1;
			if(strcmp(buffer, "/hFace_W") == 0) in_FloodCell_hFace_W = 0;
			if(strcmp(buffer, "etFace_W") == 0) in_FloodCell_etFace_W = 1;
			if(strcmp(buffer, "/etFace_W") == 0) in_FloodCell_etFace_W = 0;
			if(strcmp(buffer, "qxFace_W") == 0) in_FloodCell_qxFace_W = 1;
			if(strcmp(buffer, "/qxFace_W") == 0) in_FloodCell_qxFace_W = 0;
			if(strcmp(buffer, "qyFace_W") == 0) in_FloodCell_qyFace_W = 1;
			if(strcmp(buffer, "/qyFace_W") == 0) in_FloodCell_qyFace_W = 0;
			if(strcmp(buffer, "hFace_N") == 0) in_FloodCell_hFace_N = 1;
			if(strcmp(buffer, "/hFace_N") == 0) in_FloodCell_hFace_N = 0;
			if(strcmp(buffer, "etFace_N") == 0) in_FloodCell_etFace_N = 1;
			if(strcmp(buffer, "/etFace_N") == 0) in_FloodCell_etFace_N = 0;
			if(strcmp(buffer, "qxFace_N") == 0) in_FloodCell_qxFace_N = 1;
			if(strcmp(buffer, "/qxFace_N") == 0) in_FloodCell_qxFace_N = 0;
			if(strcmp(buffer, "qyFace_N") == 0) in_FloodCell_qyFace_N = 1;
			if(strcmp(buffer, "/qyFace_N") == 0) in_FloodCell_qyFace_N = 0;
			if(strcmp(buffer, "hFace_S") == 0) in_FloodCell_hFace_S = 1;
			if(strcmp(buffer, "/hFace_S") == 0) in_FloodCell_hFace_S = 0;
			if(strcmp(buffer, "etFace_S") == 0) in_FloodCell_etFace_S = 1;
			if(strcmp(buffer, "/etFace_S") == 0) in_FloodCell_etFace_S = 0;
			if(strcmp(buffer, "qxFace_S") == 0) in_FloodCell_qxFace_S = 1;
			if(strcmp(buffer, "/qxFace_S") == 0) in_FloodCell_qxFace_S = 0;
			if(strcmp(buffer, "qyFace_S") == 0) in_FloodCell_qyFace_S = 1;
			if(strcmp(buffer, "/qyFace_S") == 0) in_FloodCell_qyFace_S = 0;
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
            if(strcmp(buffer, "xmin") == 0) in_env_xmin = 1;
            if(strcmp(buffer, "/xmin") == 0) in_env_xmin = 0;
			if(strcmp(buffer, "xmax") == 0) in_env_xmax = 1;
            if(strcmp(buffer, "/xmax") == 0) in_env_xmax = 0;
			if(strcmp(buffer, "ymin") == 0) in_env_ymin = 1;
            if(strcmp(buffer, "/ymin") == 0) in_env_ymin = 0;
			if(strcmp(buffer, "ymax") == 0) in_env_ymax = 1;
            if(strcmp(buffer, "/ymax") == 0) in_env_ymax = 0;
			if(strcmp(buffer, "dt") == 0) in_env_dt = 1;
            if(strcmp(buffer, "/dt") == 0) in_env_dt = 0;
			if(strcmp(buffer, "DXL") == 0) in_env_DXL = 1;
            if(strcmp(buffer, "/DXL") == 0) in_env_DXL = 0;
			if(strcmp(buffer, "DYL") == 0) in_env_DYL = 1;
            if(strcmp(buffer, "/DYL") == 0) in_env_DYL = 0;
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
				if(in_FloodCell_inDomain){
                    FloodCell_inDomain = (int) fpgu_strtol(buffer); 
                }
				if(in_FloodCell_x){
                    FloodCell_x = (int) fpgu_strtol(buffer); 
                }
				if(in_FloodCell_y){
                    FloodCell_y = (int) fpgu_strtol(buffer); 
                }
				if(in_FloodCell_z0){
                    FloodCell_z0 = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_h){
                    FloodCell_h = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qx){
                    FloodCell_qx = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qy){
                    FloodCell_qy = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_timeStep){
                    FloodCell_timeStep = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_minh_loc){
                    FloodCell_minh_loc = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_E){
                    FloodCell_hFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_E){
                    FloodCell_etFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_E){
                    FloodCell_qxFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_E){
                    FloodCell_qyFace_E = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_W){
                    FloodCell_hFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_W){
                    FloodCell_etFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_W){
                    FloodCell_qxFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_W){
                    FloodCell_qyFace_W = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_N){
                    FloodCell_hFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_N){
                    FloodCell_etFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_N){
                    FloodCell_qxFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_N){
                    FloodCell_qyFace_N = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_hFace_S){
                    FloodCell_hFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_etFace_S){
                    FloodCell_etFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qxFace_S){
                    FloodCell_qxFace_S = (double) fpgu_strtod(buffer); 
                }
				if(in_FloodCell_qyFace_S){
                    FloodCell_qyFace_S = (double) fpgu_strtod(buffer); 
                }
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
            if(in_env_xmin){
              
                    env_xmin = (double) fpgu_strtod(buffer);
                    
                    set_xmin(&env_xmin);
                  
              }
            if(in_env_xmax){
              
                    env_xmax = (double) fpgu_strtod(buffer);
                    
                    set_xmax(&env_xmax);
                  
              }
            if(in_env_ymin){
              
                    env_ymin = (double) fpgu_strtod(buffer);
                    
                    set_ymin(&env_ymin);
                  
              }
            if(in_env_ymax){
              
                    env_ymax = (double) fpgu_strtod(buffer);
                    
                    set_ymax(&env_ymax);
                  
              }
            if(in_env_dt){
              
                    env_dt = (double) fpgu_strtod(buffer);
                    
                    set_dt(&env_dt);
                  
              }
            if(in_env_DXL){
              
                    env_DXL = (double) fpgu_strtod(buffer);
                    
                    set_DXL(&env_DXL);
                  
              }
            if(in_env_DYL){
              
                    env_DYL = (double) fpgu_strtod(buffer);
                    
                    set_DYL(&env_DYL);
                  
              }
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
