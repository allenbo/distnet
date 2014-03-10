/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CONV_UTIL_CUH
#define	CONV_UTIL_CUH

#include <nvmatrix.cuh>
#include <iostream>
using namespace std;

void sum(NVMatrix& src, int axis, NVMatrix& target);
void addVector(NVMatrix& taret, NVMatrix& vec);
void convLocalMaxPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int imageY, int outputsY, int outputsX);
void convLocalAvgPool(NVMatrix& images, NVMatrix& target, int numFilters,
                   int subsX, int startX, int strideX, int imageY, int outputsY, int outputsX);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY, float scaleTargets, float scaleOutput);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsY, int outputsX, int imageY, int imageX);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsy, int outputsX, int imageY, int imageX,
                      float scaleTargets, float scaleOutput);

void convResponseNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, int imageY, float addScale, float powScale);
void convResponseNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, int imageY, float addScale, float powScale, float scaleTargets, float scaleOutput);
void convContrastNorm(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, int imageY, float addScale, float powScale);
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& meanDiffs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, int imageY, float addScale, float powScale, float scaleTargets, float scaleOutput);
void convResponseNormCrossMapUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeF, int imageY, float addScale, float powScale, bool blocked, float scaleTargets, float scaleOutput);
void convResponseNormCrossMap(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeF, int imageY, float addScale,
                              float powScale, bool blocked);

void convGaussianBlur(NVMatrix& images, NVMatrix& filter, NVMatrix& target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs);
void convBedOfNails(NVMatrix& images, NVMatrix& target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput);
void convBedOfNailsUndo(NVMatrix& actsGrad, NVMatrix& target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput);

void convResizeBilinear(NVMatrix& images, NVMatrix& target, int imgSize, int tgtSize, float scale);
void convRGBToYUV(NVMatrix& images, NVMatrix& target);
void convRGBToLAB(NVMatrix& images, NVMatrix& target, bool center);
void convCrop(NVMatrix& imgs, NVMatrix& target, int imgSize, int tgtSize, int startY, int startX);
void normalizeLocalWeights(NVMatrix& weights, int numModules, float norm);
void convTICAGrad(NVMatrix& images, NVMatrix& ticas, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
void convTICA(NVMatrix& images, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
void convContrastNormCrossMap(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target,
                             int numFilters, int sizeF, int imageY, float addScale, float powScale, bool blocked);

#endif	/* CONV_UTIL_CUH */

