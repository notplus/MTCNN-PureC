#ifndef CV_H
#define CV_H

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"

#define MIN(A,B) ((A)<(B)?(A):(B))
#define MAX(A,B) ((A)>(B)?(A):(B))

struct Img
{
	unsigned char* pdata;
	int cols;
	int rows;
	int dims;
};

struct Rect
{
	int x, y;
	int width,height;
};

void CutPicture(struct Img* src, struct Img* Dst, struct Rect rec);
void RectInit(struct Rect* rec, int x, int y, int width, int height);
void Resize(struct Img* src, struct Img* Dst, int changedW, int chagedH);

#endif // !CV_H
