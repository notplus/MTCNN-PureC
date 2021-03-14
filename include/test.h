#ifndef TEST_H
#define TEST_H

#include "stdio.h"
#include "mtcnn.h"
#include "time.h"

#define WIDTH 640
#define HEIGHT 480
#define CHANNEL 3

void getInput(struct Img* image);

void getInput(struct Img* image)
{
	FILE* fp = fopen("input1.txt", "r");
	//FILE* fp = fopen("input2.txt", "r");
	image->pdata = (unsigned char*)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * CHANNEL);
	image->cols = WIDTH;
	image->rows = HEIGHT;
	image->dims = CHANNEL;

	for (size_t i = 0; i < image->cols*image->rows*image->dims; i++)
	{
		fscanf(fp, "%d ", &image->pdata[i]);
	}
}

#endif // !TEST_H

