/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMON_CUH
#define	COMMON_CUH

#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include "nvmatrix.cuh"
#include "conv_util.cuh"

enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups);
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput);

void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups);
void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups,
                     float scaleTargets, float scaleOutput);

void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput);

void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                    int moduleStride, int numImgColors, int numGroups, int sumWidth);
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, int sumWidth,
                    float scaleTargets, float scaleOutput);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                     int moduleStride, int numImgColors, int numGroups);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

#endif	/* COMMON_CUH */

