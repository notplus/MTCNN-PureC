#include "network.h"

void Image2MatrixInit(struct Img* image, struct Mat* mat)
{
	mat->channel = image->dims;
	mat->mat.numCols = image->cols;
	mat->mat.numRows = image->rows;
	mat->mat.pData = (float*)malloc(sizeof(float) * mat->channel * mat->mat.numCols * mat->mat.numRows);
}

void Image2Matrix(struct Img* image, struct Mat* mat)
{
	float* p = mat->mat.pData;
	unsigned char* pp = image->pdata;
	for (size_t i = 0; i < image->dims; i++)
	{
		for (size_t j = 0; j < image->rows; j++)
		{
			for (size_t k = 0; k < image->cols; k++)
			{
				*p = (*pp - 127.5) * 0.0078125;
				pp++;
				p++;
			}
		}
	}
}

void Im2colInit(struct Mat* input, struct Mat* matrix, struct Weight* weight)
{
	matrix->mat.numCols = ((input->mat.numCols + 2 * weight->pad - weight->kernelSize) / weight->stride + 1) *
		((input->mat.numRows + 2 * weight->pad - weight->kernelSize) / weight->stride + 1);
	matrix->mat.numRows = weight->kernelSize * weight->kernelSize * weight->lastChannel;
	matrix->channel = 1;

	matrix->mat.pData = (float*)malloc(sizeof(float) * matrix->mat.numCols * matrix->mat.numRows);
}

void Im2col(struct Mat* input, struct Mat* matrix, struct Weight* weight)
{
	int output_h = (input->mat.numRows + 2 * weight->pad - weight->kernelSize) / weight->stride + 1;
	int output_w = (input->mat.numCols + 2 * weight->pad - weight->kernelSize) / weight->stride + 1;
	int channel_size = input->mat.numRows * input->mat.numCols;

	float* p = matrix->mat.pData;
	float* pi = input->mat.pData;

	for (int channel = input->channel; channel--; pi += channel_size)
	{
		/*第二个和第三个for循环表示了输出单通道矩阵的某一列，同时体现了输出单通道矩阵的行数*/
		for (int kernel_row = 0; kernel_row < weight->kernelSize; kernel_row++)
		{
			for (int kernel_col = 0; kernel_col < weight->kernelSize; kernel_col++)
			{
				int input_row = -weight->pad + kernel_row;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
				/*第四个和第五个for循环表示了输出单通道矩阵的某一行，同时体现了输出单通道矩阵的列数*/
				for (int output_rows = output_h; output_rows; output_rows--)
				{
					if (!IS_A_GE_ZERO_AND_A_LT_B(input_row, input->mat.numRows))
					{//如果计算得到的输入图像的行值索引小于零或者大于输入图像的高(该行为pad)
						for (int output_cols = output_w; output_cols; output_cols--)
						{
							*(p++) = 0;//那么将该行在输出的矩阵上的位置置为0
						}
					}
					else
					{
						int input_col = -weight->pad + kernel_col;//在这里找到卷积核中的某一列在输入图像中的第一个操作区域的列索引
						for (int output_col = output_w; output_col; output_col--) {
							if (IS_A_GE_ZERO_AND_A_LT_B(input_col, input->mat.numCols)) {//如果计算得到的输入图像的列值索引大于等于于零或者小于输入图像的宽(该列不是pad)
								*(p++) = pi[input_row * input->mat.numCols + input_col];//将输入特征图上对应的区域放到输出矩阵上
							}
							else {//否则，计算得到的输入图像的列值索引小于零或者大于输入图像的宽(该列为pad)
								*(p++) = 0;//将该行该列在输出矩阵上的位置置为0
							}
							input_col += weight->stride;//按照宽方向步长遍历卷积核上固定列在输入图像上滑动操作的区域
						}
					}
					input_row += weight->stride;//按照高方向步长遍历卷积核上固定行在输入图像上滑动操作的区域
				}
			}
		}
	}
}

void ConvolutionInit(struct Weight* weight, struct Mat* input, struct Mat* output, struct Mat* matrix)
{
	output->channel = weight->selfChannel;
	output->mat.numRows = (input->mat.numRows + 2 * weight->pad - weight->kernelSize) / weight->stride + 1;
	output->mat.numCols = (input->mat.numCols + 2 * weight->pad - weight->kernelSize) / weight->stride + 1;
	output->mat.pData = (float*)malloc(sizeof(float) * weight->selfChannel * matrix->mat.numCols);
}

void Convolution(struct Weight* weight, struct Mat* input, struct Mat* output, struct Mat* matrix)
{
	arm_matrix_instance_f32 weight_matrix;
	weight_matrix.numCols = matrix->mat.numRows;
	weight_matrix.numRows = weight->selfChannel;
	weight_matrix.pData = weight->pdata;
	arm_mat_mult_f32(&weight_matrix, &matrix->mat, &output->mat);
}

void InitpRelu(struct pRelu* PRelu, int width)
{
	PRelu->width = width;
	PRelu->pdata = (float*)malloc(width * sizeof(float));
}

void PRelu(struct Mat* input, float* pbias, float* prelu_gmma)
{
	float* op = input->mat.pData;
	float* pb = pbias;
	float* pg = prelu_gmma;

	int dis = input->mat.numCols * input->mat.numRows;
	for (size_t i = 0; i < input->channel; i++)
	{
		for (size_t col = 0; col < dis; col++)
		{
			*op = *op + *pb;
			*op = (*op > 0) ? (*op) : (*op) * (*pg);
			op++;
		}
		pb++;
		pg++;
	}
}

void MaxPoolingInit(struct Mat* input, struct Mat* matrix, int kernelSize, int stride)
{
	matrix->mat.numCols = ceilf((float)(input->mat.numCols - kernelSize) / stride + 1);
	matrix->mat.numRows = ceilf((float)(input->mat.numRows - kernelSize) / stride + 1);
	matrix->channel = input->channel;
	matrix->mat.pData = (float*)malloc(matrix->channel * matrix->mat.numCols * matrix->mat.numRows * sizeof(float));
}

void MaxPooling(struct Mat* input, struct Mat* matrix, int kernelSize, int stride)
{
	float* p = matrix->mat.pData;
	float* pIn;
	float* ptemp;
	float maxNum = 0;
	if ((input->mat.numCols - kernelSize) % stride == 0) {
		for (int row = 0; row < matrix->mat.numRows; row++) {
			for (int col = 0; col < matrix->mat.numCols; col++) {
				pIn = input->mat.pData + row * stride * input->mat.numCols + col * stride;
				for (int channel = 0; channel < input->channel; channel++) {
					ptemp = pIn + channel * input->mat.numRows * input->mat.numCols;
					maxNum = *ptemp;
					for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
						for (int i = 0; i < kernelSize; i++) {
							if (maxNum < *(ptemp + i + kernelRow * input->mat.numCols))
								maxNum = *(ptemp + i + kernelRow * input->mat.numCols);
						}
					}
					*(p + channel * matrix->mat.numCols * matrix->mat.numRows) = maxNum;
				}
				p++;
			}
		}
	}
	else {
		int diffh = 0, diffw = 0;
		for (int channel = 0; channel < input->channel; channel++) {
			pIn = input->mat.pData + channel * input->mat.numCols * input->mat.numRows;
			for (int row = 0; row < matrix->mat.numRows; row++) {
				for (int col = 0; col < matrix->mat.numCols; col++) {
					ptemp = pIn + row * stride * input->mat.numCols + col * stride;
					maxNum = *ptemp;
					diffh = row * stride - input->mat.numRows + 1;
					diffw = col * stride - input->mat.numRows + 1;
					for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
						if ((kernelRow + diffh) > 0)break;
						for (int i = 0; i < kernelSize; i++) {
							if ((i + diffw) > 0)break;
							if (maxNum < *(ptemp + i + kernelRow * input->mat.numCols))
								maxNum = *(ptemp + i + kernelRow * input->mat.numCols);
						}
					}
					*p++ = maxNum;
				}
			}
		}
	}
}

void FullconnectInit(struct Weight* weight, struct Mat* output)
{
	output->channel = weight->selfChannel;
	output->mat.numCols = 1;
	output->mat.numRows = 1;
	output->mat.pData = (float*)malloc(sizeof(float) * weight->selfChannel);
}

void Fullconnect(struct Weight* weight, struct Mat* input, struct Mat* output)
{
	arm_matrix_instance_f32 weight_matrix;
	arm_matrix_instance_f32 input_matrix;
	weight_matrix.numCols = weight->lastChannel;
	weight_matrix.numRows = weight->selfChannel;
	weight_matrix.pData = weight->pdata;
	input_matrix.numRows = input->mat.numCols * input->mat.numRows * input->channel;
	input_matrix.numCols = 1;
	input_matrix.pData = input->mat.pData;
	output->mat.numRows = weight->selfChannel;
	arm_mat_mult_f32(&weight_matrix, &input_matrix, &output->mat);
	output->mat.numRows = 1;
}

void Softmax(const struct Mat* input)
{
	float* p2D = input->mat.pData;
	float* p3D = NULL;
	long mapSize = input->mat.numCols * input->mat.numRows;
	float eleSum = 0;
	for (int row = 0; row < input->mat.numRows; row++) {
		for (int col = 0; col < input->mat.numCols; col++) {
			eleSum = 0;
			for (int channel = 0; channel < input->channel; channel++) {
				p3D = p2D + channel * mapSize;
				*p3D = exp(*p3D);
				eleSum += *p3D;
			}
			for (int channel = 0; channel < input->channel; channel++) {
				p3D = p2D + channel * mapSize;
				*p3D = (*p3D) / eleSum;
			}
			p2D++;
		}
	}
}

void AddBias(struct Mat* input, float* pbias)
{
	float* op = input->mat.pData;
	float* pb = pbias;

	long dis = input->mat.numCols * input->mat.numRows;
	for (int channel = 0; channel < input->channel; channel++) {
		for (int col = 0; col < dis; col++) {
			*op = *op + *pb;
			op++;
		}
		pb++;
	}
}

void BubbleSort(struct VectorOrderScore* bboxScore)
{
	struct orderScore temp;
	for (size_t i = 0; i < bboxScore->size-1; i++)
	{
		for (size_t j = 0; j < bboxScore->size-i-1; j++)
		{
			if (bboxScore->data[j].score>bboxScore->data[j+1].score)
			{
				temp = bboxScore->data[j];
				bboxScore->data[j] = bboxScore->data[j+1];
				bboxScore->data[j + 1] = temp;
			}
		}
	}
}

long InitConvAndFc(struct Weight* weight, int schannel, int lchannel, int kersize, int stride, int pad)
{
	weight->selfChannel = schannel;
	weight->lastChannel = lchannel;
	weight->kernelSize = kersize;
	weight->stride = stride;
	weight->pad = pad;
	weight->pbias = (float*)malloc(schannel * sizeof(float));

	long byteLenght = weight->selfChannel * weight->lastChannel * weight->kernelSize * weight->kernelSize;
	weight->pdata = (float*)malloc(byteLenght * sizeof(float));

	return byteLenght;
}

void Nms(struct VectorBbox* boundingBox, struct VectorOrderScore* bboxScore, const float ouverlap_threshold, char modelName)
{
	if (boundingBox->size == 0)
		return;
	int* heros;
	heros = (int*)malloc(sizeof(int));
	int heros_size=0;
	int heros_memory = 1;

	BubbleSort(bboxScore);
	
	int order = 0;
	float IOU = 0;
	float maxX = 0, maxY = 0;
	float minX = 0, minY = 0;
	while (bboxScore->size>0)
	{
		order = bboxScore->data[bboxScore->size - 1].oriOrder;
		bboxScore->size--;
		if (order < 0) continue;

		{
			heros_size++;
			if (heros_memory >= heros_size)
				heros[heros_size - 1] = order;
			else
			{
				heros_memory *= 2;
				int* temp = (int*)malloc(sizeof(int) * heros_memory);
				for (size_t i = 0; i < heros_size-1; i++)
					temp[i] = heros[i];
				temp[heros_size - 1] = order;
				free(heros);
				heros = temp;
			}
		}

		boundingBox->data[order].exist = 0;
		
		for ( int num = 0; num < boundingBox->size; num++)
		{
			if (boundingBox->data[num].exist)
			{
				//the iou
				maxX = MAX(boundingBox->data[num].x1, boundingBox->data[order].x1);
				maxY = MAX(boundingBox->data[num].y1, boundingBox->data[order].y1);
				minX = MIN(boundingBox->data[num].x2, boundingBox->data[order].x2);
				minY = MIN(boundingBox->data[num].y2, boundingBox->data[order].y2);
				//maxX1 and maxY1 reuse
				maxX = MAX((minX - maxX + 1), 0);
				maxY = MAX((minY - maxY + 1), 0);
				//IOU reuse for the area of two bbox
				IOU = maxX * maxY;
				if (modelName == 'u')
					IOU = IOU / (boundingBox->data[num].area + boundingBox->data[order].area - IOU);
				else if (modelName == 'm')
					IOU = IOU / MIN(boundingBox->data[num].area, boundingBox->data[order].area);
				if (IOU>ouverlap_threshold)
				{
					boundingBox->data[num].exist = 0;
					for (int i = 0; i < bboxScore->size; i++)
					{
						if (bboxScore->data[i].oriOrder==num)
						{
							bboxScore->data[i].oriOrder = -1;
							break;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < heros_size; i++)
		boundingBox->data[heros[i]].exist = 1;
}

void ReadData(char* filename, long dataNumber[], float* pTeam[])
{
	FILE* fp = fopen(filename, "r");
	int i = 0, count = 0;
	float num;

	float* p = pTeam[0];

	while (1 == fscanf(fp, "%f\n", &num))
	{
		if (i < dataNumber[count])
		{
			*(pTeam[count])++ = num;
		}
		else
		{
			count++;
			dataNumber[count] += dataNumber[count - 1];
			*(pTeam[count])++ = num;
		}
		i++;
	}
}

void RefineAndSquareBbox(struct VectorBbox* vec_Bbox, const int height, const int width)
{
	if (vec_Bbox->size==0)
		return;

	float bbw = 0, bbh = 0, maxSide = 0;
	float h = 0, w = 0;
	float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	for (int i = 0; i < vec_Bbox->size; i++)
	{
		if (vec_Bbox->data[i].exist)
		{
			struct Bbox* temp = &vec_Bbox->data[i];
			bbh = temp->x2 - temp->x1 + 1;
			bbw = temp->y2 - temp->y1 + 1;
			x1 = temp->x1 + temp->regreCoord[1] * bbh;
			y1 = temp->y1 + temp->regreCoord[0] * bbw;
			x2 = temp->x2 + temp->regreCoord[3] * bbh;
			y2 = temp->y2 + temp->regreCoord[2] * bbw;

			h = x2 - x1 + 1;
			w = y2 - y1 + 1;

			maxSide = (h > w) ? h : w;
			x1 = x1 + h * 0.5 - maxSide * 0.5;
			y1 = y1 + w * 0.5 - maxSide * 0.5;
			temp->x2 = round(x1 + maxSide - 1);
			temp->y2 = round(y1 + maxSide - 1);
			temp->x1 = round(x1);
			temp->y1 = round(y1);

			//boundary check
			if (temp->x1 < 0) temp->x1 = 0;
			if (temp->y1 < 0) temp->y1 = 0;
			if (temp->x2 > height) temp->x2 = height - 1;
			if (temp->y2 > width) temp->y2 = width - 1;

			temp->area = (temp->x2 - temp->x1) * (temp->y2 - temp->y1);
		}
	}
}

void vector_float_push_back(struct VectorFloat* vec, float addone)
{
	vec->size++;
	if (vec->memory >= vec->size)
	{
		vec->data[vec->size - 1] = addone;
	}
	else
	{
		vec->memory *= 2;
		float* temp = (float*)malloc(sizeof(float) * vec->memory);
		memcpy(temp, vec->data, (vec->size - 1) * sizeof(float));
		temp[vec->size - 1] = addone;
		free(vec->data);
		vec->data = temp;
	}
}

void vector_Bbox_init(struct VectorBbox* bbox)
{
	bbox->data = (struct Bbox*)malloc(sizeof(struct Bbox));
	bbox->size = 0;
	bbox->memory = 1;
}

void vector_Bbox_push_back(struct VectorBbox* bbox, struct Bbox addone)
{
	bbox->size++;
	if (bbox->memory >= bbox->size)
	{
		bbox->data[bbox->size - 1] = addone;
	}
	else
	{
		bbox->memory *= 2;
		struct Bbox* temp = (struct Bbox*)malloc(sizeof(struct Bbox) * bbox->memory);
		for (int i = 0; i < bbox->size - 1; i++)
			temp[i] = bbox->data[i];
		temp[bbox->size - 1] = addone;
		free(bbox->data);
		bbox->data = temp;
	}
}

void vector_Bbox_clear(struct VectorBbox* bbox)
{
	free(bbox->data);
	bbox->memory = 0;
	bbox->size = 0;
}

void vector_orderScore_init(struct VectorOrderScore* bboxScore)
{
	bboxScore->data = (struct orderScore*)malloc(sizeof(struct orderScore));
	bboxScore->size = 0;
	bboxScore->memory = 1;
}

void vector_orderScore_push_back(struct VectorOrderScore* bboxScore, struct orderScore addone)
{
	bboxScore->size++;
	if (bboxScore->memory >= bboxScore->size)
	{
		bboxScore->data[bboxScore->size - 1] = addone;
	}
	else
	{
		bboxScore->memory *= 2;
		struct orderScore* temp = (struct orderScore*)malloc(sizeof(struct orderScore) * bboxScore->memory);
		for (int i = 0; i < bboxScore->size - 1; i++)
			temp[i] = bboxScore->data[i];
		temp[bboxScore->size - 1] = addone;
		free(bboxScore->data);
		bboxScore->data = temp;
	}
}

void vector_orderScore_clear(struct VectorOrderScore* bboxScore)
{
	free(bboxScore->data);
	bboxScore->memory = 0;
	bboxScore->size = 0;
}