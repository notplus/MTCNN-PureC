#include "cv.h"

void CutPicture(struct Img* src, struct Img* Dst, struct Rect rec)
{
	Dst->cols = rec.width;
	Dst->rows = rec.height;
	Dst->dims = src->dims;
	Dst->pdata = (unsigned char*)malloc(sizeof(unsigned char) * Dst->cols * Dst->rows * Dst->dims);

	unsigned char* p = src->pdata;
	unsigned char* pd = Dst->pdata;

	for (size_t i = 0; i < src->dims; i++)
	{
		p = src->pdata + rec.y * src->cols + rec.x + i * src->cols * src->rows;
		for (size_t j = 0; j < rec.height; j++)
		{
			memcpy(pd, p, sizeof(unsigned char) * rec.width);
			pd += rec.width;
			p += src->cols;
		}
	}
}

void RectInit(struct Rect* rec, int x, int y, int width, int height)
{
	rec->x = x;
	rec->y = y;
	rec->width = width;
	rec->height = height;
}

void Resize(struct Img* src, struct Img* dst, int changedW, int changedH)
{
	dst->pdata = (unsigned char*)malloc(sizeof(unsigned char) * changedW * changedW * src->dims);
	dst->cols = changedW;
	dst->rows = changedH;
	dst->dims = src->dims;
	double scale_x = (double)src->cols / dst->cols;
	double scale_y = (double)src->rows / dst->rows;

	unsigned char* dataDst = dst->pdata;
	int stepDst = dst->cols * 3;
	unsigned char* dataSrc = src->pdata;
	int stepSrc = src->cols * 3;
	int iWidthSrc = src->cols;
	int iHeightSrc = src->rows;

	int srcChannelSize = iWidthSrc * iHeightSrc;

	for (int j = 0; j < dst->rows; j++)
	{
		float fy = (float)((j + 0.5) * scale_y - 0.5);
		int sy = floorf(fy);
		fy -= sy;
		sy = MIN(sy, iHeightSrc - 2);
		sy = MAX(0, sy);

		short cbufy[2];
		cbufy[0] = (short)((1.f - fy) * 2048);
		cbufy[1] = 2048 - cbufy[0];
		
		for (int i = 0; i < dst->cols; i++)
		{
			float fx = (float)((i + 0.5) * scale_x - 0.5);
			int sx = floorf(fx);
			fx -= sx;

			if (sx<0)
				fx = 0, sx = 0;
			if (sx >= iWidthSrc - 1)
				fx = 0, sx = iWidthSrc - 2;

			short cbufx[2];
			cbufx[0] = (short)((1.f - fx) * 2048);
			cbufx[1] = 2048 - cbufx[0];

			for (int k = 0; k < src->dims; k++)
			{
				*(dataDst + j * dst->cols + i + k * changedH * changedW) = (
					*(dataSrc + sy * iWidthSrc + sx + k* srcChannelSize) * cbufx[0] * cbufy[0] +
					*(dataSrc + (sy + 1) * iWidthSrc + sx + k * srcChannelSize) * cbufx[0] * cbufy[1] +
					*(dataSrc + sy * iWidthSrc + (sx + 1) + k * srcChannelSize) * cbufx[1] * cbufy[0] +
					*(dataSrc + (sy + 1) * iWidthSrc + (sx + 1) + k * srcChannelSize) * cbufx[1] * cbufy[1] ) >> 22;
			}

		}

	}
}
