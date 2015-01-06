#include "StdAfx.h"
#include "KeyFrame_label.h"



KeyFrame::KeyFrame(void)
{
}


KeyFrame::~KeyFrame(void)
{
}

double KeyFrame::calDist(_Vector4f p1,_Vector4f p2)
{
	double dist=0;
	dist+=(p1.x-p2.x)*(p1.x-p2.x);
	dist+=(p1.y-p2.y)*(p1.y-p2.y);
	dist+=(p1.z-p2.z)*(p1.z-p2.z);
	dist=sqrt(dist);
	return dist;
}

double KeyFrame::calVel(vector<_Vector4f> Hands,double* vel,int InitHeight)
{
	_Vector4f lastPoint=Hands[0];
	_Vector4f currPoint;

	if(Hands[0].w < InitHeight)
		vel[0]=0;
	else
		vel[0]=-1;

	double meanVel=0;
	int count=0;
	for(int i=1;i<Hands.size();i++)
	{
		
		currPoint=Hands[i];
		if(Hands[i].w < InitHeight)
		{
			vel[i]=calDist(lastPoint,currPoint);
			meanVel+=vel[i];
			count++;
		}
		else
			vel[i]=-1;
		lastPoint=currPoint;
	}

	if(count==0)
		return -1;
	return meanVel/count;
}

bool KeyFrame::getKeyFrameLabel(vector<SLR_ST_Skeleton> SkeletonData, int leastFrameNum, int InitHeight, int* KeyFrameLabel)
{
	framesNum=SkeletonData.size();
	double meanRight, meanLeft;
	if(framesNum>0)
	{
		rightVel=new double[framesNum];
		leftVel=new double[framesNum];
		Vel=new double[framesNum];
	}
	else
		return false;

	for(int i=0;i<framesNum;i++)
	{
		_Vector4f tempLeft,tempRight;
		tempLeft.x = 1000 * SkeletonData[i]._3dPoint[7].x;
		tempLeft.y = 1000 * SkeletonData[i]._3dPoint[7].y;
		tempLeft.z = 1000 * SkeletonData[i]._3dPoint[7].z;
		tempLeft.w=SkeletonData[i]._2dPoint[7].y;
		leftHands.push_back(tempLeft);

		tempRight.x = 1000 * SkeletonData[i]._3dPoint[11].x;
		tempRight.y = 1000 * SkeletonData[i]._3dPoint[11].y;
		tempRight.z = 1000 * SkeletonData[i]._3dPoint[11].z;
		tempRight.w=SkeletonData[i]._2dPoint[11].y;
		rightHands.push_back(tempRight);

		KeyFrameLabel[i]=0;
	}

	meanRight=calVel(rightHands,rightVel,InitHeight);
	meanLeft=calVel(leftHands,leftVel,InitHeight);

	int KeyFrameCount=0;
	for(int i=0;i<framesNum;i++)
	{
		if( (rightVel[i]>=0 && rightVel[i]<meanRight) || (leftVel[i]>=0 && leftVel[i]<meanLeft) )
		{
			KeyFrameLabel[i]=1;
			KeyFrameCount++;
		}
	}
	if(KeyFrameCount >= leastFrameNum)
	{
		delete[] rightVel;
		delete[] leftVel;
		delete[] Vel;
		return true;
	}

	vector<int> leftFrames;
	for(int i=0;i<framesNum;i++)
	{
		if(KeyFrameLabel[i]==0)
		{
			if(rightVel[i]<0)
			{
				if(leftVel[i] < 0)
					Vel[i]=-1;
				else
					Vel[i]=leftVel[i];
			}
			else if(leftVel[i]<0)
				Vel[i]=rightVel[i];
			else if(leftVel[i] < rightVel[i])
				Vel[i]=leftVel[i];
			else
				Vel[i]=rightVel[i];
			if(Vel[i]>=0)
				leftFrames.push_back(i);
		}
		
	}

	if(leftFrames.size()==0)
	{
		delete[] rightVel;
		delete[] leftVel;
		delete[] Vel;
		return false;
	}
	while(KeyFrameCount < leastFrameNum)
	{
		double currMin=-1;
		int currIdx=-1;
		for(int i=0;i<leftFrames.size();i++)
		{
			if(KeyFrameLabel[leftFrames[i]]==0)
			{
				if(Vel[leftFrames[i]] < currMin || currMin==-1)
				{
					currMin=Vel[leftFrames[i]];
					currIdx=leftFrames[i];
				}
			}
		}
		if(currIdx==-1)
			break;
		else
		{
			KeyFrameLabel[currIdx]=1;
			KeyFrameCount++;
		}
	}

	
	
	delete[] rightVel;
	delete[] leftVel;
	delete[] Vel;

	if(KeyFrameCount < leastFrameNum)
		return false;
	return true;
}


