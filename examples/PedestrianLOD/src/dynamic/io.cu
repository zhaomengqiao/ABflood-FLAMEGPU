
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count)
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
    fputs("\t<EYE_X>", file);
    sprintf(data, "%f", (*get_EYE_X()));
    fputs(data, file);
    fputs("</EYE_X>\n", file);
    fputs("\t<EYE_Y>", file);
    sprintf(data, "%f", (*get_EYE_Y()));
    fputs(data, file);
    fputs("</EYE_Y>\n", file);
    fputs("\t<EYE_Z>", file);
    sprintf(data, "%f", (*get_EYE_Z()));
    fputs(data, file);
    fputs("</EYE_Z>\n", file);
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
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count)
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
    
    /* tags for environment global variables */
    int in_env;
    int in_env_TIME_SCALER;
    
    int in_env_STEER_WEIGHT;
    
    int in_env_AVOID_WEIGHT;
    
    int in_env_COLLISION_WEIGHT;
    
    int in_env_GOAL_WEIGHT;
    
    int in_env_EYE_X;
    
    int in_env_EYE_Y;
    
    int in_env_EYE_Z;
    
	/* set agent count to zero */
	*h_xmachine_memory_agent_count = 0;
	
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

    /* Variables for environment variables */
    float env_TIME_SCALER;
    float env_STEER_WEIGHT;
    float env_AVOID_WEIGHT;
    float env_COLLISION_WEIGHT;
    float env_GOAL_WEIGHT;
    float env_EYE_X;
    float env_EYE_Y;
    float env_EYE_Z;
    


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
    in_env_TIME_SCALER = 0;
    in_env_STEER_WEIGHT = 0;
    in_env_AVOID_WEIGHT = 0;
    in_env_COLLISION_WEIGHT = 0;
    in_env_GOAL_WEIGHT = 0;
    in_env_EYE_X = 0;
    in_env_EYE_Y = 0;
    in_env_EYE_Z = 0;
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

    /* Default variables for environment variables */
    env_TIME_SCALER = 0;
    env_STEER_WEIGHT = 0;
    env_AVOID_WEIGHT = 0;
    env_COLLISION_WEIGHT = 0;
    env_GOAL_WEIGHT = 0;
    env_EYE_X = 0;
    env_EYE_Y = 0;
    env_EYE_Z = 0;
    
    
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
			
            /* environment variables */
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
			if(strcmp(buffer, "EYE_X") == 0) in_env_EYE_X = 1;
            if(strcmp(buffer, "/EYE_X") == 0) in_env_EYE_X = 0;
			if(strcmp(buffer, "EYE_Y") == 0) in_env_EYE_Y = 1;
            if(strcmp(buffer, "/EYE_Y") == 0) in_env_EYE_Y = 0;
			if(strcmp(buffer, "EYE_Z") == 0) in_env_EYE_Z = 1;
            if(strcmp(buffer, "/EYE_Z") == 0) in_env_EYE_Z = 0;
			

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
				
            }
            else if (in_env){
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
            if(in_env_EYE_X){
              
                    env_EYE_X = (float) fgpu_atof(buffer);
                    
                    set_EYE_X(&env_EYE_X);
                  
              }
            if(in_env_EYE_Y){
              
                    env_EYE_Y = (float) fgpu_atof(buffer);
                    
                    set_EYE_Y(&env_EYE_Y);
                  
              }
            if(in_env_EYE_Z){
              
                    env_EYE_Z = (float) fgpu_atof(buffer);
                    
                    set_EYE_Z(&env_EYE_Z);
                  
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
