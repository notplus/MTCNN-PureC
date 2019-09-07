#include "mtcnn.h"
#include "math.h"

void InitPnet(struct Pnet* pnet)
{
	pnet->Pthreshold = 0.6;
	pnet->nms_threshold = 0.5;
	pnet->firstFlag = 1;
	pnet->rgb = (struct Mat*)malloc(sizeof(struct Mat));

	pnet->bboxScore.data = (struct orderScore*)malloc(sizeof(struct orderScore));
	pnet->bboxScore.memory = 1;
	pnet->bboxScore.size = 0;

	pnet->boundingBox.data = (struct Bbox*)malloc(sizeof(struct Bbox));
	pnet->boundingBox.memory = 1;
	pnet->boundingBox.size = 0;

	pnet->conv1_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	pnet->conv1_out = (struct Mat*)malloc(sizeof(struct Mat));
	pnet->maxPooling1 = (struct Mat*)malloc(sizeof(struct Mat));

	pnet->maxPooling_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	pnet->conv2_out = (struct Mat*)malloc(sizeof(struct Mat));

	pnet->conv3_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	pnet->conv3_out = (struct Mat*)malloc(sizeof(struct Mat));

	pnet->score_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	pnet->score = (struct Mat*)malloc(sizeof(struct Mat));

	pnet->location_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	pnet->location = (struct Mat*)malloc(sizeof(struct Mat));

	pnet->conv1_wb = (struct Weight*)malloc(sizeof(struct Weight));
	pnet->prelu1 = (struct pRelu*)malloc(sizeof(struct pRelu));
	pnet->conv2_wb = (struct Weight*)malloc(sizeof(struct Weight));
	pnet->prelu2 = (struct pRelu*)malloc(sizeof(struct pRelu));
	pnet->conv3_wb = (struct Weight*)malloc(sizeof(struct Weight));
	pnet->prelu3 = (struct pRelu*)malloc(sizeof(struct pRelu));
	pnet->conv4c1_wb = (struct Weight*)malloc(sizeof(struct Weight));
	pnet->conv4c2_wb = (struct Weight*)malloc(sizeof(struct Weight));

	long conv1_out = InitConvAndFc(pnet->conv1_wb, 10, 3, 3, 1, 0);
	InitpRelu(pnet->prelu1, 10);
	long conv2_out = InitConvAndFc(pnet->conv2_wb, 16, 10, 3, 1, 0);
	InitpRelu(pnet->prelu2, 16);
	long conv3_out = InitConvAndFc(pnet->conv3_wb, 32, 16, 3, 1, 0);
	InitpRelu(pnet->prelu3, 32);
	long conv4c1 = InitConvAndFc(pnet->conv4c1_wb, 2, 32, 1, 1, 0);
	long conv4c2 = InitConvAndFc(pnet->conv4c2_wb, 4, 32, 1, 1, 0);
	long dataNumber[13] = { conv1_out,10,10, conv2_out,16,16, conv3_out,32,32, conv4c1,2, conv4c2,4 };
	float* pointTeam[13] = { pnet->conv1_wb->pdata, pnet->conv1_wb->pbias, pnet->prelu1->pdata,
							pnet->conv2_wb->pdata, pnet->conv2_wb->pbias, pnet->prelu2->pdata,
							pnet->conv3_wb->pdata, pnet->conv3_wb->pbias, pnet->prelu3->pdata,
							pnet->conv4c1_wb->pdata, pnet->conv4c1_wb->pbias,
							pnet->conv4c2_wb->pdata, pnet->conv4c2_wb->pbias
	};

	char filename[9] = "Pnet.txt";
	ReadData(filename, dataNumber, pointTeam);
}

void RunPnet(struct Img* image, float scale, struct Pnet* pnet)
{
	if (pnet->firstFlag)
	{
		Image2MatrixInit(image, pnet->rgb);

		Im2colInit(pnet->rgb, pnet->conv1_matrix, pnet->conv1_wb);
		ConvolutionInit(pnet->conv1_wb, pnet->rgb, pnet->conv1_out, pnet->conv1_matrix);

		MaxPoolingInit(pnet->conv1_out, pnet->maxPooling1, 2, 2);
		Im2colInit(pnet->maxPooling1, pnet->maxPooling_matrix, pnet->conv2_wb);
		ConvolutionInit(pnet->conv2_wb, pnet->maxPooling1, pnet->conv2_out, pnet->maxPooling_matrix);

		Im2colInit(pnet->conv2_out, pnet->conv3_matrix, pnet->conv3_wb);
		ConvolutionInit(pnet->conv3_wb, pnet->conv2_out, pnet->conv3_out, pnet->conv3_matrix);

		Im2colInit(pnet->conv3_out, pnet->score_matrix, pnet->conv4c1_wb);
		ConvolutionInit(pnet->conv4c1_wb, pnet->conv3_out, pnet->score, pnet->score_matrix);

		Im2colInit(pnet->conv3_out, pnet->location_matrix, pnet->conv4c2_wb);
		ConvolutionInit(pnet->conv4c2_wb, pnet->conv3_out, pnet->location, pnet->location_matrix);
		pnet->firstFlag = 0;
	}

	Image2Matrix(image, pnet->rgb);
	Im2col(pnet->rgb, pnet->conv1_matrix, pnet->conv1_wb);
	Convolution(pnet->conv1_wb, pnet->rgb, pnet->conv1_out, pnet->conv1_matrix);
	PRelu(pnet->conv1_out, pnet->conv1_wb->pbias, pnet->prelu1->pdata);
	MaxPooling(pnet->conv1_out, pnet->maxPooling1, 2, 2);

	Im2col(pnet->maxPooling1, pnet->maxPooling_matrix, pnet->conv2_wb);
	Convolution(pnet->conv2_wb, pnet->maxPooling1, pnet->conv2_out, pnet->maxPooling_matrix);
	PRelu(pnet->conv2_out, pnet->conv2_wb->pbias, pnet->prelu2->pdata);

	Im2col(pnet->conv2_out, pnet->conv3_matrix, pnet->conv3_wb);
	Convolution(pnet->conv3_wb, pnet->conv2_out, pnet->conv3_out, pnet->conv3_matrix);
	PRelu(pnet->conv3_out, pnet->conv3_wb->pbias, pnet->prelu3->pdata);

	Im2col(pnet->conv3_out, pnet->score_matrix, pnet->conv4c1_wb);
	Convolution(pnet->conv4c1_wb, pnet->conv3_out, pnet->score, pnet->score_matrix);
	AddBias(pnet->score, pnet->conv4c1_wb->pbias);
	Softmax(pnet->score);

	Im2col(pnet->conv3_out, pnet->location_matrix, pnet->conv4c2_wb);
	Convolution(pnet->conv4c2_wb, pnet->conv3_out, pnet->location, pnet->location_matrix);
	AddBias(pnet->location, pnet->conv4c2_wb->pbias);

	GenerateBbox(pnet->score, pnet->location, scale, pnet);
}

void GenerateBbox(struct Mat* score, struct Mat* location, float scale, struct Pnet* pnet)
{
	int stride = 2;
	int cellsize = 12;
	int count = 0;

	float* p = score->mat.pData + score->mat.numCols * score->mat.numRows;
	float* plocal = location->mat.pData;
	struct Bbox bbox;
	struct orderScore order;
	for (size_t row = 0; row < score->mat.numRows; row++)
	{
		for (size_t col = 0; col < score->mat.numCols; col++)
		{
			if (*p > pnet->Pthreshold)
			{
				bbox.score = *p;
				order.score = *p;
				order.oriOrder = count;
				bbox.x1 = roundf((stride * row + 1) / scale);
				bbox.y1 = roundf((stride * col + 1) / scale);
				bbox.x2 = roundf((stride * row + 1 + cellsize) / scale);
				bbox.y2 = roundf((stride * col + 1 + cellsize) / scale);
				bbox.exist = 1;
				bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
				for (size_t channel = 0; channel < 4; channel++)
					bbox.regreCoord[channel] = *(plocal + channel * location->mat.numCols * location->mat.numRows);
				vector_Bbox_push_back(&pnet->boundingBox, bbox);
				vector_orderScore_push_back(&pnet->bboxScore, order);
				count++;
			}
			p++;
			plocal++;
		}
	}
}

void InitRnet(struct Rnet* rnet)
{
	rnet->Rthreshold = 0.7;

	rnet->rgb = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->conv1_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->conv1_out = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->pooling1_out = (struct Mat*)malloc(sizeof(struct Mat));

	rnet->conv2_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->conv2_out = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->pooling2_out = (struct Mat*)malloc(sizeof(struct Mat));

	rnet->conv3_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->conv3_out = (struct Mat*)malloc(sizeof(struct Mat));

	rnet->fc4_out = (struct Mat*)malloc(sizeof(struct Mat));

	rnet->score = (struct Mat*)malloc(sizeof(struct Mat));
	rnet->location = (struct Mat*)malloc(sizeof(struct Mat));

	rnet->conv1_wb = (struct Weight*)malloc(sizeof(struct Weight));
	rnet->prelu1 = (struct pRelu*)malloc(sizeof(struct pRelu));
	rnet->conv2_wb = (struct Weight*)malloc(sizeof(struct Weight));
	rnet->prelu2 = (struct pRelu*)malloc(sizeof(struct pRelu));
	rnet->conv3_wb = (struct Weight*)malloc(sizeof(struct Weight));
	rnet->prelu3 = (struct pRelu*)malloc(sizeof(struct pRelu));
	rnet->fc4_wb = (struct Weight*)malloc(sizeof(struct Weight));
	rnet->prelu4 = (struct pRelu*)malloc(sizeof(struct pRelu));
	rnet->score_wb = (struct Weight*)malloc(sizeof(struct Weight));
	rnet->location_wb = (struct Weight*)malloc(sizeof(struct Weight));

	long conv1_out = InitConvAndFc(rnet->conv1_wb, 28, 3, 3, 1, 0);
	InitpRelu(rnet->prelu1, 28);
	long conv2_out = InitConvAndFc(rnet->conv2_wb, 48, 28, 3, 1, 0);
	InitpRelu(rnet->prelu2, 48);
	long conv3_out = InitConvAndFc(rnet->conv3_wb, 64, 48, 2, 1, 0);
	InitpRelu(rnet->prelu3, 64);
	long fc4 = InitConvAndFc(rnet->fc4_wb, 128, 576, 1, 1, 0);
	InitpRelu(rnet->prelu4, 128);
	long score = InitConvAndFc(rnet->score_wb, 2, 128, 1, 1, 0);
	long location = InitConvAndFc(rnet->location_wb, 4, 128, 1, 1, 0);
	long dataNumber[16] = { conv1_out,28,28, conv2_out,48,48, conv3_out,64,64, fc4,128,128, score,2, location,4 };
	float* pointTeam[16] = { rnet->conv1_wb->pdata, rnet->conv1_wb->pbias, rnet->prelu1->pdata,
							rnet->conv2_wb->pdata, rnet->conv2_wb->pbias, rnet->prelu2->pdata,
							rnet->conv3_wb->pdata, rnet->conv3_wb->pbias, rnet->prelu3->pdata,
							rnet->fc4_wb->pdata, rnet->fc4_wb->pbias, rnet->prelu4->pdata,
							rnet->score_wb->pdata, rnet->score_wb->pbias,
							rnet->location_wb->pdata, rnet->location_wb->pbias
	};
	char filename[9] = "Rnet.txt";
	ReadData(filename, dataNumber, pointTeam);

	RnetImage2MatrixInit(rnet->rgb);
	Im2colInit(rnet->rgb, rnet->conv1_matrix, rnet->conv1_wb);
	ConvolutionInit(rnet->conv1_wb, rnet->rgb, rnet->conv1_out, rnet->conv1_matrix);
	MaxPoolingInit(rnet->conv1_out, rnet->pooling1_out, 3, 2);
	Im2colInit(rnet->pooling1_out, rnet->conv2_matrix, rnet->conv2_wb);
	ConvolutionInit(rnet->conv2_wb, rnet->pooling1_out, rnet->conv2_out, rnet->conv2_matrix);
	MaxPoolingInit(rnet->conv2_out, rnet->pooling2_out, 3, 2);
	Im2colInit(rnet->pooling2_out, rnet->conv3_matrix, rnet->conv3_wb);
	ConvolutionInit(rnet->conv3_wb, rnet->pooling2_out, rnet->conv3_out, rnet->conv3_matrix);
	FullconnectInit(rnet->fc4_wb, rnet->fc4_out);
	FullconnectInit(rnet->score_wb, rnet->score);
	FullconnectInit(rnet->location_wb, rnet->location);
}

void RunRnet(struct Img* image, struct Rnet* rnet)
{
	Image2Matrix(image, rnet->rgb);

	Im2col(rnet->rgb, rnet->conv1_matrix, rnet->conv1_wb);
	Convolution(rnet->conv1_wb, rnet->rgb, rnet->conv1_out, rnet->conv1_matrix);
	PRelu(rnet->conv1_out, rnet->conv1_wb->pbias, rnet->prelu1->pdata);
	MaxPooling(rnet->conv1_out, rnet->pooling1_out, 3, 2);

	Im2col(rnet->pooling1_out, rnet->conv2_matrix, rnet->conv2_wb);
	Convolution(rnet->conv2_wb, rnet->pooling1_out, rnet->conv2_out, rnet->conv2_matrix);
	PRelu(rnet->conv2_out, rnet->conv2_wb->pbias, rnet->prelu2->pdata);
	MaxPooling(rnet->conv2_out, rnet->pooling2_out, 3, 2);

	Im2col(rnet->pooling2_out, rnet->conv3_matrix, rnet->conv3_wb);
	Convolution(rnet->conv3_wb, rnet->pooling2_out, rnet->conv3_out, rnet->conv3_matrix);
	PRelu(rnet->conv3_out, rnet->conv3_wb->pbias, rnet->prelu3->pdata);

	Fullconnect(rnet->fc4_wb, rnet->conv3_out, rnet->fc4_out);
	PRelu(rnet->fc4_out, rnet->fc4_wb->pbias, rnet->prelu4->pdata);

	Fullconnect(rnet->score_wb, rnet->fc4_out, rnet->score);
	AddBias(rnet->score, rnet->score_wb->pbias);
	Softmax(rnet->score);

	Fullconnect(rnet->location_wb, rnet->fc4_out, rnet->location);
	AddBias(rnet->location, rnet->location_wb->pbias);
}

void RnetImage2MatrixInit(struct Mat* input)
{
	input->channel = 3;
	input->mat.numCols = 24;
	input->mat.numRows = 24;

	input->mat.pData = (float*)malloc(sizeof(float) * 24 * 24 * 3);
}

void InitOnet(struct Onet* onet)
{
	onet->Othreshold = 0.8;
	onet->rgb = (struct Mat*)malloc(sizeof(struct Mat));

	onet->conv1_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	onet->conv1_out = (struct Mat*)malloc(sizeof(struct Mat));
	onet->pooling1_out = (struct Mat*)malloc(sizeof(struct Mat));

	onet->conv2_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	onet->conv2_out = (struct Mat*)malloc(sizeof(struct Mat));
	onet->pooling2_out = (struct Mat*)malloc(sizeof(struct Mat));

	onet->conv3_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	onet->conv3_out = (struct Mat*)malloc(sizeof(struct Mat));
	onet->pooling3_out = (struct Mat*)malloc(sizeof(struct Mat));

	onet->conv4_matrix = (struct Mat*)malloc(sizeof(struct Mat));
	onet->conv4_out = (struct Mat*)malloc(sizeof(struct Mat));

	onet->fc5_out = (struct Mat*)malloc(sizeof(struct Mat));

	onet->score = (struct Mat*)malloc(sizeof(struct Mat));
	onet->location = (struct Mat*)malloc(sizeof(struct Mat));
	onet->keyPoint = (struct Mat*)malloc(sizeof(struct Mat));

	onet->conv1_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->prelu1 = (struct pRelu*)malloc(sizeof(struct pRelu));
	onet->conv2_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->prelu2 = (struct pRelu*)malloc(sizeof(struct pRelu));
	onet->conv3_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->prelu3 = (struct pRelu*)malloc(sizeof(struct pRelu));
	onet->conv4_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->prelu4 = (struct pRelu*)malloc(sizeof(struct pRelu));
	onet->fc5_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->prelu5 = (struct pRelu*)malloc(sizeof(struct pRelu));
	onet->score_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->location_wb = (struct Weight*)malloc(sizeof(struct Weight));
	onet->keyPoint_wb = (struct Weight*)malloc(sizeof(struct Weight));

	// //                             w        sc  lc ks s  p
	long conv1_out = InitConvAndFc(onet->conv1_wb, 32, 3, 3, 1, 0);
	InitpRelu(onet->prelu1, 32);
	long conv2_out = InitConvAndFc(onet->conv2_wb, 64, 32, 3, 1, 0);
	InitpRelu(onet->prelu2, 64);
	long conv3_out = InitConvAndFc(onet->conv3_wb, 64, 64, 3, 1, 0);
	InitpRelu(onet->prelu3, 64);
	long conv4 = InitConvAndFc(onet->conv4_wb, 128, 64, 2, 1, 0);
	InitpRelu(onet->prelu4, 128);
	long fc5 = InitConvAndFc(onet->fc5_wb, 256, 1152, 1, 1, 0);
	InitpRelu(onet->prelu5, 256);
	long score = InitConvAndFc(onet->score_wb, 2, 256, 1, 1, 0);
	long location = InitConvAndFc(onet->location_wb, 4, 256, 1, 1, 0);
	long keyPoint = InitConvAndFc(onet->keyPoint_wb, 10, 256, 1, 1, 0);
	long dataNumber[21] = { conv1_out,32,32, conv2_out,64,64, conv3_out,64,64, conv4,128,128, fc5,256,256, score,2, location,4, keyPoint,10 };
	float* pointTeam[21] = { onet->conv1_wb->pdata, onet->conv1_wb->pbias, onet->prelu1->pdata,
								onet->conv2_wb->pdata, onet->conv2_wb->pbias, onet->prelu2->pdata,
								onet->conv3_wb->pdata, onet->conv3_wb->pbias, onet->prelu3->pdata,
								onet->conv4_wb->pdata, onet->conv4_wb->pbias, onet->prelu4->pdata,
								onet->fc5_wb->pdata, onet->fc5_wb->pbias, onet->prelu5->pdata,
								onet->score_wb->pdata, onet->score_wb->pbias,
								onet->location_wb->pdata, onet->location_wb->pbias,
								onet->keyPoint_wb->pdata, onet->keyPoint_wb->pbias
	};
	char filename[9] = "Onet.txt";
	ReadData(filename, dataNumber, pointTeam);

	OnetImage2MatrixInit(onet->rgb);

	Im2colInit(onet->rgb, onet->conv1_matrix, onet->conv1_wb);
	ConvolutionInit(onet->conv1_wb, onet->rgb, onet->conv1_out, onet->conv1_matrix);
	MaxPoolingInit(onet->conv1_out, onet->pooling1_out, 3, 2);

	Im2colInit(onet->pooling1_out, onet->conv2_matrix, onet->conv2_wb);
	ConvolutionInit(onet->conv2_wb, onet->pooling1_out, onet->conv2_out, onet->conv2_matrix);
	MaxPoolingInit(onet->conv2_out, onet->pooling2_out, 3, 2);

	Im2colInit(onet->pooling2_out, onet->conv3_matrix, onet->conv3_wb);
	ConvolutionInit(onet->conv3_wb, onet->pooling2_out, onet->conv3_out, onet->conv3_matrix);
	MaxPoolingInit(onet->conv3_out, onet->pooling3_out, 2, 2);

	Im2colInit(onet->pooling3_out, onet->conv4_matrix, onet->conv4_wb);
	ConvolutionInit(onet->conv4_wb, onet->pooling3_out, onet->conv4_out, onet->conv4_matrix);

	FullconnectInit(onet->fc5_wb, onet->fc5_out);
	FullconnectInit(onet->score_wb, onet->score);
	FullconnectInit(onet->location_wb, onet->location);
	FullconnectInit(onet->keyPoint_wb, onet->keyPoint);
}

void RunOnet(struct Img* image, struct Onet* onet)
{
	Image2Matrix(image, onet->rgb);

	Im2col(onet->rgb, onet->conv1_matrix, onet->conv1_wb);
	Convolution(onet->conv1_wb, onet->rgb, onet->conv1_out, onet->conv1_matrix);
	PRelu(onet->conv1_out, onet->conv1_wb->pbias, onet->prelu1->pdata);
	MaxPooling(onet->conv1_out, onet->pooling1_out, 3, 2);

	Im2col(onet->pooling1_out, onet->conv2_matrix, onet->conv2_wb);
	Convolution(onet->conv2_wb, onet->pooling1_out, onet->conv2_out, onet->conv2_matrix);
	PRelu(onet->conv2_out, onet->conv2_wb->pbias, onet->prelu2->pdata);
	MaxPooling(onet->conv2_out, onet->pooling2_out, 3, 2);

	Im2col(onet->pooling2_out, onet->conv3_matrix, onet->conv3_wb);
	Convolution(onet->conv3_wb, onet->pooling2_out, onet->conv3_out, onet->conv3_matrix);
	PRelu(onet->conv3_out, onet->conv3_wb->pbias, onet->prelu3->pdata);
	MaxPooling(onet->conv3_out, onet->pooling3_out, 2, 2);

	//conv4
	Im2col(onet->pooling3_out, onet->conv4_matrix, onet->conv4_wb);
	Convolution(onet->conv4_wb, onet->pooling3_out, onet->conv4_out, onet->conv4_matrix);
	PRelu(onet->conv4_out, onet->conv4_wb->pbias, onet->prelu4->pdata);

	Fullconnect(onet->fc5_wb, onet->conv4_out, onet->fc5_out);
	PRelu(onet->fc5_out, onet->fc5_wb->pbias, onet->prelu5->pdata);

	//conv6_1   score
	Fullconnect(onet->score_wb, onet->fc5_out, onet->score);
	AddBias(onet->score, onet->score_wb->pbias);
	Softmax(onet->score);

	//conv6_2   location
	Fullconnect(onet->location_wb, onet->fc5_out, onet->location);
	AddBias(onet->location, onet->location_wb->pbias);

	//conv6_2   location
	Fullconnect(onet->keyPoint_wb, onet->fc5_out, onet->keyPoint);
	AddBias(onet->keyPoint, onet->keyPoint_wb->pbias);
}

void OnetImage2MatrixInit(struct Mat* input)
{
	input->channel = 3;
	input->mat.numCols = 48;
	input->mat.numRows = 48;
	input->mat.pData = (float*)malloc(sizeof(float) * 48 * 48 * 3);
}

void InitMtcnn(struct Mtcnn* network, int row, int col)
{
	network->nms_threshold[0] = 0.7;
	network->nms_threshold[1] = 0.7;
	network->nms_threshold[2] = 0.7;

	float minl = min(row, col);
	int MIN_DET_SIZE = 12;
	int minsize = 60;
	float m = (float)MIN_DET_SIZE / minsize;
	minl *= m;
	float factor = 0.709;
	int factor_count = 0;

	network->scales = (struct VectorFloat*)malloc(sizeof(struct VectorFloat));
	network->scales->data = (float*)malloc(sizeof(float));
	network->scales->memory = 1;
	network->scales->size = 0;

	vector_Bbox_init(&network->firstBbox);
	vector_orderScore_init(&network->firstOrderScore);
	vector_Bbox_init(&network->secondBbox);
	vector_orderScore_init(&network->secondOrderScore);
	vector_Bbox_init(&network->thirdBbox);
	vector_orderScore_init(&network->thirdOrderScore);

	while (minl > MIN_DET_SIZE)
	{
		if (factor_count > 0)
		{
			m *= factor;
		}
		vector_float_push_back(network->scales, m);
		minl *= factor;
		factor_count++;
	}

	network->simpleFace = (struct Pnet*)malloc(sizeof(struct Pnet) * network->scales->size);
	for (size_t i = 0; i < network->scales->size; i++)
	{
		InitPnet(&network->simpleFace[i]);
	}

	InitRnet(&network->refineNet);
	InitOnet(&network->outNet);
}

void FindFace(struct Img* image, struct Mtcnn* mtcnn)
{
	struct orderScore order;
	int count = 0;
	for (size_t i = 0; i < mtcnn->scales->size; i++)
	{
		int changedH = (int)ceil(image->rows * mtcnn->scales->data[i]);
		int changedW = (int)ceil(image->cols * mtcnn->scales->data[i]);
		mtcnn->reImage= (struct Img*)malloc(sizeof(struct Img));
		Resize(image, mtcnn->reImage, changedW, changedH);
		RunPnet(mtcnn->reImage, mtcnn->scales->data[i], &mtcnn->simpleFace[i]);
		Nms(&mtcnn->simpleFace[i].boundingBox, &mtcnn->simpleFace[i].bboxScore, mtcnn->simpleFace[i].nms_threshold,'u');

		for (int j = 0; j < mtcnn->simpleFace[i].boundingBox.size; j++)
		{
			if (mtcnn->simpleFace[i].boundingBox.data[j].exist)
			{
				vector_Bbox_push_back(&mtcnn->firstBbox, mtcnn->simpleFace[i].boundingBox.data[j]);
				order.score = mtcnn->simpleFace[i].boundingBox.data[j].score;
				order.oriOrder = count;
				vector_orderScore_push_back(&mtcnn->firstOrderScore, order);
				count++;
			}
		}
		vector_Bbox_clear(&mtcnn->simpleFace[i].boundingBox);
		vector_orderScore_clear(&mtcnn->simpleFace[i].bboxScore);
	}

	//the first stage's Nms
	if (count < 1) return;
	Nms(&mtcnn->firstBbox, &mtcnn->firstOrderScore, mtcnn->nms_threshold[0],'u');
	RefineAndSquareBbox(&mtcnn->firstBbox, image->rows, image->cols);

	//second stage
	count = 0;
	for (size_t i = 0; i < mtcnn->firstBbox.size; i++)
	{
		if (mtcnn->firstBbox.data[i].exist)
		{
			struct Bbox* t = &mtcnn->firstBbox.data[i];
			struct Rect temp;
			RectInit(&temp, t->y1, t->x1, t->y2 - t->y1, t->x2 - t->x1);
			struct Img secImage, imgTemp;
			CutPicture(image, &imgTemp, temp);
			Resize(&imgTemp, &secImage, 24, 24);
			RunRnet(&secImage, &mtcnn->refineNet);
			if (*(mtcnn->refineNet.score->mat.pData+1)>mtcnn->refineNet.Rthreshold)
			{
				memcpy(t->regreCoord, mtcnn->refineNet.location->mat.pData, 4 * sizeof(float));
				t->area = (t->x2 - t->x1) * (t->y2 - t->y1);
				t->score = *(mtcnn->refineNet.score->mat.pData + 1);
				vector_Bbox_push_back(&mtcnn->secondBbox, *t);
				order.score = t->score;
				order.oriOrder = count++;
				vector_orderScore_push_back(&mtcnn->secondOrderScore, order);
			}
			else
			{
				mtcnn->firstBbox.data[i].exist = 0;
			}
		}
	}
	if (count < 1) return;
	Nms(&mtcnn->secondBbox, &mtcnn->secondOrderScore, mtcnn->nms_threshold[1],'u');
	RefineAndSquareBbox(&mtcnn->secondBbox, image->rows, image->cols);

	//third stage
	count = 0;
	for (size_t i = 0; i < mtcnn->secondBbox.size; i++)
	{
		struct Bbox* t = &mtcnn->secondBbox.data[i];
		if (t->exist)
		{
			struct Rect tempRect;
			RectInit(&tempRect, t->y1, t->x1, t->y2 - t->y1, t->x2 - t->x1);
			struct Img tempImg, thirdImage;
			CutPicture(image, &tempImg, tempRect);
			Resize(&tempImg, &thirdImage, 48, 48);
			RunOnet(&thirdImage, &mtcnn->outNet);
			float* pp = NULL;
			if (*(mtcnn->outNet.score->mat.pData+1)>mtcnn->outNet.Othreshold)
			{
				memcpy(t->regreCoord, mtcnn->outNet.location->mat.pData, 4 * sizeof(float));
				t->area = (t->x2 - t->x1) * (t->y2 - t->y1);
				t->score = *(mtcnn->outNet.score->mat.pData + 1);
				pp = mtcnn->outNet.keyPoint->mat.pData;
				for (size_t num = 0; num < 5; num++)
				{
					(t->ppoint)[num] = t->y1 + (t->y2 - t->y1) * (*(pp + num));
				}
				for (size_t num = 0; num < 5; num++)
				{
					(t->ppoint)[num + 5] = t->x1 + (t->x2 - t->x1) * (*(pp + num + 5));
				}
				vector_Bbox_push_back(&mtcnn->thirdBbox, *t);
				order.score = t->score;
				order.oriOrder = count++;
				vector_orderScore_push_back(&mtcnn->thirdOrderScore, order);
			}
			else
			{
				t->exist = 0;
			}
		}
	}

	if (count < 1) return;
	RefineAndSquareBbox(&mtcnn->thirdBbox, image->rows, image->cols);
	Nms(&mtcnn->thirdBbox, &mtcnn->thirdOrderScore, mtcnn->nms_threshold[2], 'm');
	
	for (size_t i = 0; i < mtcnn->thirdBbox.size; i++)
	{
		struct Bbox* t = &mtcnn->thirdBbox.data[i];
		if (t->exist)
		{
			printf("boundingBox%d is %d %d %d %d\n", i, t->y1, t->x1, t->y2, t->x2);
			for (size_t num = 0; num < 5; num++)
			{
				printf("keyPoint%d is %f %f\n", num, t->ppoint[num], t->ppoint[num + 5]);
			}
			printf("------------------------\n");
		}
	}

	vector_Bbox_clear(&mtcnn->firstBbox);
	vector_orderScore_clear(&mtcnn->firstOrderScore);
	vector_Bbox_clear(&mtcnn->secondBbox);
	vector_orderScore_clear(&mtcnn->secondOrderScore);
	vector_Bbox_clear(&mtcnn->thirdBbox);
	vector_orderScore_clear(&mtcnn->thirdOrderScore);
}
