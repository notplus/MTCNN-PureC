#include "mtcnn.h"
#include "test.h"

int main()
{
	struct Img image;
	getInput(&image);
	struct Mtcnn network;
	InitMtcnn(&network,image.rows,image.cols);

	clock_t start, end;
	start = clock();
	FindFace(&image, &network);
	end = clock();

	printf("time is %fms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);

	return 0;
}