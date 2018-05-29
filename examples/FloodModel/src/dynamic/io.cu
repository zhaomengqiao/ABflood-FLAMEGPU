
  
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

unsigned long int fpgu_strtoul(const char* str){
    return strtoul(str, NULL, 0);
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
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_FloodCell_list* h_FloodCells_Default, xmachine_memory_FloodCell_list* d_FloodCells_Default, int h_xmachine_memory_FloodCell_Default_count)
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
    
    fputs("\t<dt>", file);
    sprintf(data, "%f", *get_dt());
    fputs(data, file);
    fputs("</dt>\n", file);
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
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_FloodCell_list* h_FloodCells, int* h_xmachine_memory_FloodCell_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name;
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
    
    /* tags for environment global variables */
    int in_env;
    int in_env_dt;
    
	/* set agent count to zero */
	*h_xmachine_memory_FloodCell_count = 0;
	
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

    /* Variables for environment variables */
    double env_dt;
    


	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
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
    in_env_dt = 0;
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

    /* Default variables for environment variables */
    env_dt = 0;
    
    
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
		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

		/* If the end of a tag */
		if(c == '>')
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
			
            /* environment variables */
            if(strcmp(buffer, "dt") == 0) in_env_dt = 1;
            if(strcmp(buffer, "/dt") == 0) in_env_dt = 0;
			

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
				
            }
            else if (in_env){
            if(in_env_dt){
              
                  //scalar value input
                  env_dt = (double) fpgu_strtod(buffer);
                  set_dt(&env_dt);
                  
              }
            
          }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
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

	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

