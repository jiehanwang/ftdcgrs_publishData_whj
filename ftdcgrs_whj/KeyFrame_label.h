#pragma once
#include "globalDefine.h"
#include<vector>
using namespace std;


class KeyFrame
{
	int framesNum;
	double* rightVel;
	double* leftVel;
	double* Vel;
	vector<_Vector4f> rightHands;
	vector<_Vector4f> leftHands; 
	double calVel(vector<_Vector4f> Hands,double* vel,int InitHeight);
	double calDist(_Vector4f p1,_Vector4f p2);
public:
	bool getKeyFrameLabel(vector<SLR_ST_Skeleton> SkeletonData, int leastFrameNum, int InitHeight, int* KeyFrameLabel);
	KeyFrame(void);
	~KeyFrame(void);
};

