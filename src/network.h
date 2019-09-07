#define _CRT_SECURE_NO_WARNINGS
#ifndef NETWORK_H
#define NETWORK_H

#include "stdlib.h"
#include "arm_math.h"
#include "cv.h"
#include "stdio.h"
#include "string.h"

#define IS_A_GE_ZERO_AND_A_LT_B(a, b) (a >= 0 && a < b) ? 1 : 0

struct Mat
{
	arm_matrix_instance_f32 mat;
	int channel;
};

struct pRelu
{
	float* pdata;
	int width;
};

struct Weight
{
	float* pdata;
	float* pbias;
	int lastChannel;
	int selfChannel;
	int kernelSize;
	int stride;
	int pad;
};

struct Bbox
{
	float score;
	int x1, y1;
	int x2, y2;
	float area;
	int exist;
	float ppoint[10];
	float regreCoord[4];
};

struct orderScore
{
	float score;
	int oriOrder;
};

struct VectorFloat
{
	float* data;
	int size;
	int memory;
};

struct VectorBbox
{
	struct Bbox* data;
	int size;
	int memory;
};

struct VectorOrderScore
{
	struct orderScore* data;
	int size;
	int memory;
};

void Image2MatrixInit(struct Img* image, struct Mat* mat);
void Image2Matrix(struct Img* image, struct Mat* mat);
void Im2colInit(struct Mat* input, struct Mat* matrix, struct Weight* weight);
void Im2col(struct Mat* input, struct Mat* matrix, struct Weight* weight);
void ConvolutionInit(struct Weight* weight, struct Mat* input, struct Mat* output, struct Mat* matrix);
void Convolution(struct Weight* weight, struct Mat* input, struct Mat* output, struct Mat* matrix);
void InitpRelu(struct pRelu* PRelu, int width);
void PRelu(struct Mat* input, float* pbias, float* prelu_gmma);
void MaxPoolingInit(struct Mat* input, struct Mat* Matrix, int kernelSize, int stride);
void MaxPooling(struct Mat* input, struct Mat* matrix, int kernelSize, int stride);
void FullconnectInit(struct Weight* weight, struct Mat* output);
void Fullconnect(struct Weight* weight, struct Mat* input, struct Mat* output);
void Softmax(const struct Mat* input);

void AddBias(struct Mat* mat, float* pbias);
void BubbleSort(struct VectorOrderScore* bboxScore);
long InitConvAndFc(struct Weight* weight, int schannel, int lchannel, int kersize, int stride, int pad);
void Nms(struct VectorBbox* boundingBox, struct VectorOrderScore* bboxScore, const float ouverlap_threshold, char modelName);
void ReadData(char* filename, long dataNumber[], float* pTeam[]);
void RefineAndSquareBbox(struct VectorBbox* vec_Bbox, const int height, const int weight);

void vector_float_push_back(struct VectorFloat* vec, float addone);
void vector_Bbox_init(struct VectorBbox* bbox);
void vector_Bbox_push_back(struct VectorBbox* bbox, struct Bbox addone);
void vector_Bbox_clear(struct VectorBbox* bbox);
void vector_orderScore_init(struct VectorOrderScore* bboxScore);
void vector_orderScore_push_back(struct VectorOrderScore* bboxScore, struct orderScore addone);
void vector_orderScore_clear(struct VectorOrderScore* bboxScore);

#endif // NETWORK_H