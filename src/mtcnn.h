#ifndef MTCNN_H
#define MTCNN_H

#include "network.h"

struct Pnet
{
	float nms_threshold;
	float Pthreshold;
	int firstFlag;
	struct VectorBbox boundingBox;
	struct VectorOrderScore bboxScore;

	struct Mat* rgb;
	struct Mat* conv1_matrix;
	
	struct Mat* conv1_out;
	struct Mat* maxPooling1;
	struct Mat* maxPooling_matrix;

	struct Mat* conv2_out;
	struct Mat* conv3_matrix;

	struct Mat* conv3_out;
	struct Mat* score_matrix;

	struct Mat* score;
	struct Mat* location_matrix;
	struct Mat* location;

	//weight
	struct Weight* conv1_wb;
	struct pRelu* prelu1;
	struct Weight* conv2_wb;
	struct pRelu* prelu2;
	struct Weight* conv3_wb;
	struct pRelu* prelu3;
	struct Weight* conv4c1_wb;
	struct Weight* conv4c2_wb;
	
};

struct Rnet
{
	float Rthreshold;
	struct Mat* score;
	struct Mat* location;

	struct Mat* rgb;
	struct Mat* conv1_matrix;
	struct Mat* conv1_out;
	struct Mat* pooling1_out;

	struct Mat* conv2_matrix;
	struct Mat* conv2_out;
	struct Mat* pooling2_out;

	struct Mat* conv3_matrix;
	struct Mat* conv3_out;

	struct Mat* fc4_out;

	//weight
	struct Weight* conv1_wb;
	struct pRelu* prelu1;
	struct Weight* conv2_wb;
	struct pRelu* prelu2;
	struct Weight* conv3_wb;
	struct pRelu* prelu3;
	
	struct Weight* fc4_wb;
	struct pRelu* prelu4;
	struct Weight* score_wb;
	struct Weight* location_wb;
};

struct Onet
{
	float Othreshold;
	struct Mat* score;
	struct Mat* location;
	struct Mat* keyPoint;

	struct Mat* rgb;
	struct Mat* conv1_matrix;
	struct Mat* conv1_out;
	struct Mat* pooling1_out;

	struct Mat* conv2_matrix;
	struct Mat* conv2_out;
	struct Mat* pooling2_out;

	struct Mat* conv3_matrix;
	struct Mat* conv3_out;
	struct Mat* pooling3_out;

	struct Mat* conv4_matrix;
	struct Mat* conv4_out;

	struct Mat* fc5_out;

	//weight
	struct Weight* conv1_wb;
	struct pRelu* prelu1;
	struct Weight* conv2_wb;
	struct pRelu* prelu2;
	struct Weight* conv3_wb;
	struct pRelu* prelu3;
	struct Weight* conv4_wb;
	struct pRelu* prelu4;
	struct Weight* fc5_wb;
	struct pRelu* prelu5;
	struct Weight* score_wb;
	struct Weight* location_wb;
	struct Weight* keyPoint_wb;

};

struct Mtcnn
{
	struct Img* reImage;
	float nms_threshold[3];
	struct VectorFloat* scales;
	struct Pnet* simpleFace;
	struct VectorBbox firstBbox;
	struct VectorOrderScore firstOrderScore;
	struct Rnet refineNet;
	struct VectorBbox secondBbox;
	struct VectorOrderScore secondOrderScore;
	struct Onet outNet;
	struct VectorBbox thirdBbox;
	struct VectorOrderScore thirdOrderScore;
};

void InitPnet(struct Pnet* pnet);
void RunPnet(struct Img* image, float scale, struct Pnet* pnet);
void GenerateBbox(struct Mat* score, struct Mat* location, float scale, struct Pnet* pnet);

void InitRnet(struct Rnet* rnet);
void RunRnet(struct Img* image, struct Rnet* rnet);
void RnetImage2MatrixInit(struct Mat* input);

void InitOnet(struct Onet* onet);
void RunOnet(struct Img* image, struct Onet* onet);
void OnetImage2MatrixInit(struct Mat* input);

void InitMtcnn(struct Mtcnn* network, int row, int col);
void FindFace(struct Img* image,struct Mtcnn* mtcnn);

#endif // !MTCNN_H
