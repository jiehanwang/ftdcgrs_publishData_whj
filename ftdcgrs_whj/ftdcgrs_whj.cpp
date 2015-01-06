// ftdcgrs_whj.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ftdcgrs_whj.h"
#include "Hmm.h"
#include "time.h"
#include "Recognition.h"
#include <iostream>
#include "Readvideo.h"
#include "HandSegment.h"
#include "KeyFrame_label.h"
#include <direct.h>
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//#define TReadAndFeaGene
//#define TskpReadAndFeaGene
//#define ColorReadAndFeaGene
//#define GenerateDataForTrainAndTest       //Step 1

//#define OneM_mergeFeature
#define OneM_OutputReorganize
#define OneM_TRAIN
#define OneM_TEST



//// #define OutputReorganize                  //Step 2
//// #define TRAIN                             //Step 3
//// #define TEST                              //Step 4
//// #define mergeFeature



#define PostureSIZE 64
#define OriHOGSize 648
#define ReduceHOGsize  100
#define desHOGSize 51
// The one and only application object

CWinApp theApp;

using namespace std;

vector<float> myPCA[OriHOGSize];  

bool GetHOGHistogram_Patch(IplImage *img,vector<double> &hog_hist);//取得图像img的HOG特征向量

void mergeKeyFrame(vector<int> &flags, int mergeSpace, int deleteSegNum);

//bool GetHOGHistogram_OriDepth(Mat handMat,vector<double> &hog_hist);

void GetLBPfeature(IplImage* image, vector<float> &lbpFeature);

bool HandFeatureRetrieve(IplImage* image,SLR_ST_Skeleton skeletonData,vector<double> &hog_hist, CString leftHandFileName, CString rightHandFileName
	,CvRect leftHand, CvRect rightHand);

void LBP (IplImage *src,IplImage *dst);

void readstr(FILE *f,char *string)
{
	do
	{
		fgets(string, 1000, f);
	} while ((string[0] == '/') || (string[0] == '\n'));
	return;
}

void readInPCAMatrix(void)
{

	//Read in the ground truth.
	FILE *filein;
	char oneline[1000];
	filein = fopen("..\\input\\idx_100.txt", "rt");    // File To Load World Data From
	for (int loop = 0; loop < OriHOGSize; loop++)
	{
		//sentenceLength[loop] = 0;
		readstr(filein,oneline);
		char* sp = oneline; 
		float num; 
		int read; 
		while( sscanf(sp, "%f %n", &num, &read)!=EOF )
		{ 
			//printf("%d\t", num); 
			myPCA[loop].push_back(num);
			//sentenceLength[loop]++;
			//groundTruth[loop].push_back(num);
			sp += read-1; 
		} 
	}
	fclose(filein);
}

void featureReduce(vector<double> srcFeature, vector<double> &desFeature)
{
	for (int i=0; i<desHOGSize/*ReduceHOGsize*/; i++)
	{
		double temp = 0.0;
		for (int j=0; j<OriHOGSize; j++)
		{
			temp += srcFeature[j]*myPCA[j][i]; 
		}
		desFeature.push_back(temp);
	}
}

void mergeHMMFeatures_train(CString file1,CString file2,CString file3, CString file4, CString dstFile)
{
	fstream fw(dstFile,ios::out);
	int frame,feaNums;
	float featureNum;
	fstream f1(file1,ios::in);
	f1 >> frame >> feaNums;
	fw << frame << " " << feaNums << endl;
	int i, j;
	for(i=0; i<frame; i++)
	{
		for(j=0; j<feaNums; j++)
		{
			f1 >> featureNum;
			fw << featureNum << " ";
		}
		fw << endl;
	}
	f1.close();

	fstream f2(file2,ios::in);
	f2 >> frame >> feaNums;
	fw << frame << " " << feaNums << endl;
	for(i=0; i<frame; i++)
	{
		for(j=0; j<feaNums; j++)
		{
			f2 >> featureNum;
			fw << featureNum << " ";
		}
		fw << endl;
	}
	f2.close();

	fstream f3(file3,ios::in);
	f3 >> frame >> feaNums;
	fw << frame << " " << feaNums << endl;
	for(i=0; i<frame; i++)
	{
		for(j=0; j<feaNums; j++)
		{
			f3 >> featureNum;
			fw << featureNum << " ";
		}
		fw << endl;
	}
	f3.close();

	fstream f4(file4,ios::in);
	f4 >> frame >> feaNums;
	fw << frame << " " << feaNums << endl;
	for(i=0; i<frame; i++)
	{
		for(j=0; j<feaNums; j++)
		{
			f4 >> featureNum;
			fw << featureNum << " ";
		}
		fw << endl;
	}
	f4.close();

	fw.close();
}

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
//////////////////////////////////////////////////////////////////////////
//Settings
//////////////////////////////////////////////////////////////////////////

	CString dir = "..\\output\\dim61_CTskp_allFrame_2000sign_12group\\";              //Step 1
	_mkdir(dir);

	//Choose the groups                                                         //Step 2                                       
	const int GroupNum = 12;    
	int groupID[GroupNum];

	groupID[0] = 1;
	groupID[1] = 1;
	groupID[2] = 2;
	groupID[3] = 2;
	groupID[4] = 3;
	groupID[5] = 3;
	groupID[6] = 4;
	groupID[7] = 4;
	groupID[8] = 5;
	groupID[9] = 6;
	groupID[10] = 7;
	groupID[11] = 8;
  
	int subgroupID[GroupNum];
	subgroupID[0] = 1;
	subgroupID[1] = 2;
	subgroupID[2] = 1;
	subgroupID[3] = 2;
	subgroupID[4] = 1;
	subgroupID[5] = 2;
	subgroupID[6] = 1;
	subgroupID[7] = 2;
	subgroupID[8] = 1;
	subgroupID[9] = 1;
	subgroupID[10] = 1;
	subgroupID[11] = 1;


	CString signListName = "..\\input\\词汇_2000.txt";       //Step 3   词汇_2000_debug


	//修改HmmStruct下的测试个数和维度                                 //Step 4

	bool bFullFrame = false;                                      //Step 5
	bool bAllFrame = true;                               
	bool bKeyFrame = false;

	int trainNum = 8;                                            //Step 6

//////////////////////////////////////////////////////////////////////////
//Prepare the data
//////////////////////////////////////////////////////////////////////////

#ifdef TskpReadAndFeaGene
	int videoTotalNum;
	BOOL videoFindFlag;
	BOOL imageFindFlag;
	CFileFind videoFileFind;
	CFileFind imageFileFind;
	ofstream outfile;
	CString fileFolderName;

	for (int myGroupID=0; myGroupID<GroupNum; myGroupID++)
	{
		int folderID = groupID[myGroupID];
		int folderID_sub = subgroupID[myGroupID];

		fileFolderName.Format("I:\\hanjie.wang_backup\\iData\\isolatedWord\\fullFrame_2000sign_12group\\dimX_T15_fullFrame\\P%02d_%d\\*.txt",folderID, folderID_sub);

		videoFindFlag = TRUE;
		videoFindFlag = videoFileFind.FindFile(fileFolderName);

		CString out_filefolder;
		out_filefolder.Format(dir + "P%02d_%d",folderID, folderID_sub);
		_mkdir(out_filefolder);

		videoTotalNum = 0;
		while(videoFindFlag)
		{ 
			videoFindFlag = videoFileFind.FindNextFile();
			CString videoFileName = videoFileFind.GetFilePath();
			CString videoName = videoFileName.Right(8).Left(4);
			videoName.Format("P%02d_%d_"+videoName,folderID, folderID_sub);
			cout<<videoName<<endl;

			int allFrame = 0;
			int dimNoUse = 0;
			vector<float> Tdata[15];
			fstream TdataFile(videoFileName,ios::in);
			TdataFile>>allFrame>>dimNoUse;

			for (int i=0; i<allFrame; i++)
			{
				for (int j=0; j<15; j++)
				{
					float temp;
					TdataFile>>temp;
					Tdata[j].push_back(temp);
				}
			}
				//All frame flag
			vector<int>   STFlag;
			float heightLimit = max(Tdata[7][0],Tdata[13][0]) + 0.055;
			int stFlameNum = 0;
			for (int i=0; i<allFrame; i++)
			{
				int tempFlag;
				if (max(Tdata[7][i],Tdata[13][i])>heightLimit)
				{
					tempFlag = 1;
					stFlameNum++;
				}
				else
				{
					tempFlag = 0;
				}
				STFlag.push_back(tempFlag);
			}

				//Read in the mask. Add key frame flag
			vector<int> flag;
			int frameNum = 0;
			int length = videoFileName.Trim().GetLength();
			CString txtFileName = videoFileName.Left(length-15);    //13
			txtFileName += "_mask\\" + videoName + ".txt";
			fstream flagFile(txtFileName,ios::in);
			flagFile>>frameNum;
			int keyFrameNum = 0;
			for (int i=0; i<frameNum; i++)
			{
				int temp;
				flagFile>>temp;
				flag.push_back(temp);
			}
			mergeKeyFrame(flag,1,2);
			for (int i=0; i<flag.size(); i++)
			{
				if (flag[i]==1) keyFrameNum++;
			}
			//////////////////////////////////////////////////////////////////////////
			//Tskp
			CString out_fileName;
			out_fileName = out_filefolder + "\\" + videoName + ".txt";
			outfile.open(out_fileName, ios::out);

			//--------------
			if (stFlameNum < 5)
			{
				bAllFrame = false;
				bFullFrame = true;
				bKeyFrame = false;
			}
			else
			{
				bAllFrame = true;
				bFullFrame = false;
				bKeyFrame = false;
			}

			//---------------


			if (bKeyFrame) outfile<<keyFrameNum<<" "<<10<<endl;
			if (bAllFrame) outfile<<stFlameNum<<" "<<10<<endl;
			//if (bAllFrame) outfile<<stFlameNum/3<<" "<<10<<endl;
			if (bFullFrame) outfile<<allFrame<<" "<<10<<endl;
			//int count = 0;
			for (int i=0; i<allFrame; i++)
			{
				if ((flag[i] == 1 && bKeyFrame) || (STFlag[i] == 1 && bAllFrame) || bFullFrame)
				{
					//if (count%3 == 0)
					{
						Vector<float> skP;
						for (int sk=0; sk<5; sk++)
						{
							for (int sk2=sk+1; sk2<5; sk2++)
							{
								float temp = pow((Tdata[sk*3+0][i]-Tdata[sk2*3+0][i]),2) 
									+ pow((Tdata[sk*3+1][i]-Tdata[sk2*3+1][i]),2)
									+ pow((Tdata[sk*3+2][i]-Tdata[sk2*3+2][i]),2);
								skP.push_back(temp);
							}
						}
						float maxSKP = 0.0;
						for (int k=0; k<skP.size(); k++)
						{
							if (maxSKP < skP[k])
							{
								maxSKP = skP[k];
							}
						}
						for (int k=0; k<skP.size(); k++)
						{
							skP[k] /=maxSKP;
							outfile<<skP[k]<<" ";
						}
						outfile<<endl;
					}
					//count++;
		
				}
				
			}

			outfile.close();
		}
	}
#endif

#ifdef TReadAndFeaGene
	int videoTotalNum;
	BOOL videoFindFlag;
	BOOL imageFindFlag;
	CFileFind videoFileFind;
	CFileFind imageFileFind;
	ofstream outfile;
	CString fileFolderName;

	for (int myGroupID=0; myGroupID<GroupNum; myGroupID++)
	{
		int folderID = groupID[myGroupID];

		fileFolderName.Format("H:\\hanjie.wang_backup\\iData\\isolatedWord\\fullFrame_1000sign_zeng\\dimX_T15_fullFrame\\P%d\\*.txt",folderID);

		videoFindFlag = TRUE;
		videoFindFlag = videoFileFind.FindFile(fileFolderName);

		CString out_filefolder;
		out_filefolder.Format(dir + "P%d",folderID);
		_mkdir(out_filefolder);

		videoTotalNum = 0;
		while(videoFindFlag)
		{ 
			videoFindFlag = videoFileFind.FindNextFile();
			CString videoFileName = videoFileFind.GetFilePath();
			CString videoName = videoFileName.Right(8).Left(4);
			videoName.Format("P%d_"+videoName,folderID);
			cout<<videoName<<endl;

			int allFrame = 0;
			int dimNoUse = 0;
			vector<float> Tdata[15];
			fstream TdataFile(videoFileName,ios::in);
			TdataFile>>allFrame>>dimNoUse;

			for (int i=0; i<allFrame; i++)
			{
				for (int j=0; j<15; j++)
				{
					float temp;
					TdataFile>>temp;
					Tdata[j].push_back(temp);
				}
			}
			//All frame flag
			vector<int>   STFlag;
			float heightLimit = max(Tdata[7][0],Tdata[13][0]) + 0.055;
			int stFlameNum = 0;
			for (int i=0; i<allFrame; i++)
			{
				int tempFlag;
				if (max(Tdata[7][i],Tdata[13][i])>heightLimit)
				{
					tempFlag = 1;
					stFlameNum++;
				}
				else
				{
					tempFlag = 0;
				}
				STFlag.push_back(tempFlag);
			}

			//Read in the mask. Add key frame flag
			vector<int> flag;
			int frameNum = 0;
			int length = videoFileName.Trim().GetLength();
			CString txtFileName = videoFileName.Left(length-13);
			txtFileName += "_mask\\" + videoName + ".txt";
			fstream flagFile(txtFileName,ios::in);
			flagFile>>frameNum;
			int keyFrameNum = 0;
			for (int i=0; i<frameNum; i++)
			{
				int temp;
				flagFile>>temp;
				flag.push_back(temp);
			}
			mergeKeyFrame(flag,1,2);
			for (int i=0; i<flag.size(); i++)
			{
				if (flag[i]==1) keyFrameNum++;
			}
			//////////////////////////////////////////////////////////////////////////
			//T
			CString out_fileName;
			out_fileName = out_filefolder + "\\" + videoName + ".txt";
			outfile.open(out_fileName, ios::out);

			//--------------
			if (stFlameNum < 5)
			{
				bAllFrame = false;
				bFullFrame = true;
				bKeyFrame = false;
			}

			//---------------

#ifndef output6T
			if (bKeyFrame) outfile<<keyFrameNum<<" "<<15<<endl;
			if (bAllFrame) outfile<<stFlameNum<<" "<<15<<endl;
			if (bFullFrame) outfile<<allFrame<<" "<<15<<endl;
#endif
#ifdef output6T
			if (bKeyFrame) outfile<<keyFrameNum<<" "<<6<<endl;
			if (bAllFrame) outfile<<stFlameNum<<" "<<6<<endl;
			if (bFullFrame) outfile<<allFrame<<" "<<6<<endl;
#endif
			for (int i=0; i<allFrame; i++)
			{
				if ((flag[i] == 1 && bKeyFrame) || (STFlag[i] == 1 && bAllFrame) || bFullFrame)
				{
#ifndef output6T
					for (int sk=0; sk<5; sk++)
					{
						outfile<<Tdata[sk*3+0][i]<<" "<<Tdata[sk*3+1][i]<<" "<<Tdata[sk*3+2][i]<<" ";
					}
#endif
#ifdef output6T
					for (int sk=2; sk<=4; sk+=2)
					{
						outfile<<Tdata[sk*3+0][i]<<" "<<Tdata[sk*3+1][i]<<" "<<Tdata[sk*3+2][i]<<" ";
					}
#endif
					
					outfile<<endl;
				}

			}

			outfile.close();
		}
	}
#endif

#ifdef ColorReadAndFeaGene

	int videoTotalNum;
	BOOL videoFindFlag;
	BOOL imageFindFlag;
	CFileFind videoFileFind;
	CFileFind imageFileFind;
	ofstream outfile;
	CString fileFolderName;
	CHandSegment handSegmentVideo;
	handSegmentVideo.init();

	for (int myGroupID=0; myGroupID<GroupNum; myGroupID++)
	{
		int folderID = groupID[myGroupID];
		int subFolderID = subgroupID[myGroupID];////////////////For dataset with subIndex
	
		fileFolderName.Format("I:\\hanjie.wang_backup\\iData\\isolatedWord\\fullFrame_2000sign_12group\\dimX_C_fullFrame\\P%02d_%d_image\\*.img",
			folderID, subFolderID);

		videoFindFlag = TRUE;
		videoFindFlag = videoFileFind.FindFile(fileFolderName);

		CString out_filefolder;
		out_filefolder.Format(dir + "P%02d_%d",folderID, subFolderID);///////////////// Format(dir + "P%d",folderID)
		_mkdir(out_filefolder);

		videoTotalNum = 0;
		while(videoFindFlag)
		{ 
			videoFindFlag = videoFileFind.FindNextFile();
			CString videoFileName = videoFileFind.GetFilePath();
			CString videoName = videoFileName.Right(8).Left(4);
			videoName.Format("P%02d_%d_"+videoName,folderID, subFolderID);////////////////////////////Format("P%d_"+videoName,folderID)
			cout<<videoName<<endl;

				//Read in "key frame" mask. Function provided by Fan Yin.
			vector<int>   flag;
			int frameNum = 0;
			int length = videoFileName.Trim().GetLength();
			//CString txtFileName = videoFileName.Left(length-18);    ///Used for one-digit file ID
			//CString txtFileName = videoFileName.Left(length-19);  ///Used for double-digit file ID 
			CString txtFileName = videoFileName.Left(length-21);    /////////////////////////////////////////
			txtFileName += "_mask\\" + videoName + ".txt";
			fstream flagFile(txtFileName,ios::in);
			flagFile>>frameNum;
			for (int i=0; i<frameNum; i++)
			{
				int temp;
				flagFile>>temp;
				flag.push_back(temp);
			}
			flagFile.close();
			mergeKeyFrame(flag,1,2);    //1.Merge the close segments. 2. Delete the isolated segments.

				//Read in the "all frame" mask. 
				//Draw a red line according to the skeleton and set the limited height.
			vector<float> STdata[2];  //Left and right hand Y values. 3D space. 
			vector<int>   STFlag;
			CString stFileName = videoFileName.Left(length-21);//////////////////////////////////////////
			stFileName += "\\" + videoName + ".txt";
			fstream stFile(stFileName,ios::in);
			int dimTemp;
			stFile>>frameNum>>dimTemp;
			for (int i=0; i<frameNum; i++)
			{
				for (int j=0; j<15; j++)
				{
					float temp;
					stFile>>temp;
					if (j == 7) STdata[0].push_back(temp);
					if (j == 13)STdata[1].push_back(temp);
				}
				
			}
			stFile.close();
			float heightLimit = max(STdata[0][0],STdata[1][0]) + 0.055;  //0.055 is tested by experiment. 
			int stFlameNum = 0;
			for (int i=0; i<frameNum; i++)
			{
				int tempFlag;
				if (max(STdata[0][i],STdata[1][i])>heightLimit)
				{
					tempFlag = 1;
					stFlameNum++;
				}
				else
				{
					tempFlag = 0;
				}
				STFlag.push_back(tempFlag);
			}

			//--------------
			if (stFlameNum < 5)
			{
				bAllFrame = false;
				bFullFrame = true;
				bKeyFrame = false;
			}
			else
			{
				bAllFrame = true;
				bFullFrame = false;
				bKeyFrame = false;
			}

			//---------------

	
				//Prepare for the output and input.
			CString out_fileName;

#ifndef outputForHTK
			out_fileName = out_filefolder + "\\" + videoName + ".txt";
#endif
			
#ifdef outputForHTK
			out_fileName.Format(out_filefolder + "\\w"+videoName.Left(8).Right(4)+"_%d.txt",folderID);
#endif
			

			outfile.open(out_fileName, ios::out);


			CString imageSearchName = videoFileName+"\\*.bmp";
			imageFindFlag = TRUE;
			imageFindFlag = imageFileFind.FindFile(imageSearchName);
			vector<Posture> vPosture;

			CString imageFileName_left;
			int count = 0;

				//The loop to find all the images in one video. 
			while (imageFindFlag)
			{
				imageFindFlag = imageFileFind.FindNextFile();
				CString imageFileName = imageFileFind.GetFilePath();
				CString imageName = imageFileName.Right(10);
				CString LR = imageName.Right(5).Left(1);
				
				if (!strcmp(LR,"L"))
				{
					imageFileName_left = imageFileName;
				}
				else if (!strcmp(LR,"R"))
				{
					if ((flag[count] == 1 && bKeyFrame) || (STFlag[count] == 1 && bAllFrame) || bFullFrame)
					{
						Posture myPostrue;

							//Load images. Further process can be done here. 
						IplImage* leftHand = cvLoadImage(imageFileName_left);
						IplImage* rightHand = cvLoadImage(imageFileName);
						if (leftHand!=NULL)
						{ 
							//IplImage* lbpImage = cvCreateImage(cvGetSize(leftHand),leftHand->depth,leftHand->nChannels);
							//LBP(leftHand,leftHand);
							//cvSaveImage("..\\test.jpg",lbpImage);
							myPostrue.leftHandImg = leftHand;
							
						}
						else
						{
							myPostrue.leftHandImg = NULL;
						}
						if (rightHand!=NULL)
						{
							//LBP(rightHand,rightHand);
							myPostrue.rightHandImg = rightHand;
						}
						else
						{
							myPostrue.rightHandImg = NULL;
						}	
						vPosture.push_back(myPostrue);
					}
					count++;

				}
				
			}
			
				//Output the HOG for hand posture.
			outfile<<vPosture.size()<<" "<<DES_FEA_NUM<<endl;
			//outfile<<vPosture.size()<<" "<<125<<endl;
			//outfile<<vPosture.size()/3<<" "<<DES_FEA_NUM<<endl;
			for(int i=0; i<vPosture.size(); i++)
			{
				Posture posture = vPosture[i];
				if(posture.leftHandImg == NULL || posture.rightHandImg == NULL)
				{
					for(int j=i-1; j>=0; j--)
					{
						Posture tmpPosture = vPosture[j];
						if(posture.leftHandImg == NULL && tmpPosture.leftHandImg != NULL)
						{
							posture.leftHandImg = cvCreateImage(cvGetSize(tmpPosture.leftHandImg),tmpPosture.leftHandImg->depth,tmpPosture.leftHandImg->nChannels);
							cvCopy(tmpPosture.leftHandImg,posture.leftHandImg);
						}
						if(posture.rightHandImg == NULL && tmpPosture.rightHandImg != NULL)
						{
							posture.rightHandImg = cvCreateImage(cvGetSize(tmpPosture.rightHandImg),tmpPosture.rightHandImg->depth,tmpPosture.rightHandImg->nChannels);
							cvCopy(tmpPosture.rightHandImg,posture.rightHandImg);
						}
						if(posture.rightHandImg != NULL && posture.leftHandImg != NULL)
						{
							break;
						}
					}
				}

				double hog_color[DES_FEA_NUM];
				handSegmentVideo.getHogFeature(posture.leftHandImg, posture.rightHandImg,hog_color);
// 				vector<float> lbpFeature_left;
// 				if (posture.leftHandImg == NULL)
// 				{
// 					for (int i=0; i<DIMENSION; i++)
// 					{
// 						outfile<<0<<" ";
// 					}
// 				}
// 				else
// 				{
// 					GetLBPfeature(posture.leftHandImg,lbpFeature_left);
// 					for (int i=0; i<lbpFeature_left.size(); i++)
// 					{
// 						outfile<<lbpFeature_left[i]<<" ";
// 					}
// 				}
				
				
				for (int d=0; d<DES_FEA_NUM; d++)
				{
					outfile<<hog_color[d]<<" ";
				}
				outfile<<endl;

			}

				//Release resource.
			for (int i=0; i<vPosture.size(); i++)
			{
				cvReleaseImage(&vPosture[i].leftHandImg);
				cvReleaseImage(&vPosture[i].rightHandImg);
			}
			vPosture.clear();
			outfile.close();
		}
	}

#endif

#ifdef GenerateDataForTrainAndTest
	readInPCAMatrix();
	
	int videoTotalNum;
	BOOL videoFindFlag;
	CFileFind videoFileFind;
	ofstream outfile;
	ofstream outfileMask;
	ofstream outfileAll;
	ofstream outUsefulLength;
	CString fileFolderName;
	CHandSegment handSegmentVideo;
	handSegmentVideo.init();


	clock_t start_video, finish_video;
	clock_t start_feature, finish_feature;
	float timeForReadvideo = 0.0;
	float timeForFeatureExtraction = 0.0;
	int totalFrameNum = 0;
	Readvideo myReadVideo;

	//for (int folderID=folderStart; folderID<folderEnd; folderID++)
	for (int myGroupID=0; myGroupID<1/*GroupNum*/; myGroupID++)
	{
		int folderID = groupID[myGroupID];
		fileFolderName.Format("H:\\hanjie.wang\\P%02d\\*.oni",folderID);
		
		videoFindFlag = TRUE;
		videoFindFlag = videoFileFind.FindFile(fileFolderName);

			//Output file folders for feature and images.
		CString out_filefolder;
		out_filefolder.Format(dir + "P%02d",folderID); ///////////////////////////////////////////////////UPdate!!!!!!
		_mkdir(out_filefolder);

		CString out_imgFileFolderDir;
		out_imgFileFolderDir = out_filefolder + "_image";
		_mkdir(out_imgFileFolderDir);

		CString out_DepthImgFileFolderDir;
		out_DepthImgFileFolderDir = out_filefolder + "_imageDepth";
		_mkdir(out_DepthImgFileFolderDir);

		CString out_MaskFileFolderDir;
		out_MaskFileFolderDir = out_filefolder + "_mask";
		_mkdir(out_MaskFileFolderDir);


		videoTotalNum = 0;
		while(videoFindFlag)
		{ 
			videoFindFlag = videoFileFind.FindNextFile();
			CString videoFileName = videoFileFind.GetFilePath();
				//Read the video including RGB, D, skeleton
			string s = (LPCTSTR)videoFileName;
			start_video = clock();
			myReadVideo.readvideo(s);
			finish_video = clock();
			timeForReadvideo += (float)(finish_video - start_video);
			int frameSize = myReadVideo.vColorData.size();

				//Get the video name from the path name. 
			//CString videoName = videoFileName.Right(25).Left(8);
			CString videoName = videoFileName.Right(21).Left(4);//25,8
			videoName.Format("P%02d_"+videoName,folderID);   //////////////////////////////////////////////UPdate!!!!!!
			//cvNamedWindow(videoName,1);

				//Output files for feature and images. 
			CString out_fileName;
			out_fileName = out_filefolder + "\\" + videoName + ".txt";
			outfile.open(out_fileName, ios::out);
			CString out_imgFileFolder;
			out_imgFileFolder.Format(out_imgFileFolderDir + "\\%s.img",videoName );
			_mkdir(out_imgFileFolder);
			CString out_DepthImgFileFolder;
			out_DepthImgFileFolder.Format(out_DepthImgFileFolderDir + "\\%s.img",videoName );
			_mkdir(out_DepthImgFileFolder);
			CString out_MaskFileName;
			out_MaskFileName.Format(out_MaskFileFolderDir + "\\%s.txt",videoName );
			outfileMask.open(out_MaskFileName, ios::out);

				//Obtain the limited height
			int heightLimit = min(myReadVideo.vSkeletonData[0]._2dPoint[7].y,
				myReadVideo.vSkeletonData[0]._2dPoint[11].y) - 20;

				//Key frame selection
			KeyFrame myKeyFrame;
			int* KeyFrameLabel;
			KeyFrameLabel = new int[frameSize];
			myKeyFrame.getKeyFrameLabel(myReadVideo.vSkeletonData,15,heightLimit,KeyFrameLabel);
			//for (int i=0; i<frameSize; i++) KeyFrameLabel[i] = 1;     //Enable this code to ignore the key frame selection

				//To get the useful frame numbers. The key frame judge can be in here.
			int framSize_Real = 0;
			outfileMask<<frameSize<<endl;
			for (int i=0; i<frameSize; i++)
			{
				int heightThisLimit = min(myReadVideo.vSkeletonData[i]._2dPoint[7].y,
					myReadVideo.vSkeletonData[i]._2dPoint[11].y);
				if (heightThisLimit < heightLimit && KeyFrameLabel[i] == 1)
				{
					framSize_Real++;
					outfileMask<<1<<endl;
				}
				else
				{
					outfileMask<<0<<endl;
				}
			}
			outfileMask.close();

			outUsefulLength.open(out_filefolder+"_RealFrame.csv",ios::app | ios::out);
			outUsefulLength<<videoName<<","<<framSize_Real<<endl;
			outUsefulLength.close();

			//outfile<<framSize_Real<<" "<<desHOGSize/*+6*/<<endl;
			outfile<<frameSize<<" "<<15<<endl;
			CvPoint headPoint, lPoint2, rPoint2;
			//vector<Posture> vPosture; 

			for (int i=0; i<frameSize; i++)
			{
					//Face detection, executed only once. 
				if (i == 0)
				{
					headPoint.x = myReadVideo.vSkeletonData[i]._2dPoint[3].x;
					headPoint.y = myReadVideo.vSkeletonData[i]._2dPoint[3].y;
					bool bHeadFound = handSegmentVideo.headDetection(
						myReadVideo.vColorData[i],
						myReadVideo.vDepthData[i],
						headPoint);

					if(bHeadFound)
					{
						handSegmentVideo.colorClusterCv(handSegmentVideo.m_pHeadImage,3);
						handSegmentVideo.getFaceNeckRegion(myReadVideo.vColorData[i],myReadVideo.vDepthData[i]);
						handSegmentVideo.copyDepthMat(myReadVideo.vDepthData[i].clone());
					}
				}

				int heightThisLimit = min(myReadVideo.vSkeletonData[i]._2dPoint[7].y,
					myReadVideo.vSkeletonData[i]._2dPoint[11].y);
				//if (heightThisLimit<heightLimit && KeyFrameLabel[i] == 1)
				{
					start_feature = clock();
					//////////////////////////////////////////////////////////////////////////
					//Color Image
					Posture posture;
					
					lPoint2.x = myReadVideo.vSkeletonData[i]._2dPoint[7].x;
					lPoint2.y = myReadVideo.vSkeletonData[i]._2dPoint[7].y;

					rPoint2.x = myReadVideo.vSkeletonData[i]._2dPoint[11].x;
					rPoint2.y = myReadVideo.vSkeletonData[i]._2dPoint[11].y;
					
					CvRect leftHand;
					CvRect rightHand;
					
					handSegmentVideo.kickHandsAll(myReadVideo.vColorData[i],myReadVideo.vDepthData[i]
					,lPoint2,rPoint2,posture,leftHand,rightHand);

					if(rightHand.x<0 || rightHand.y<0 || rightHand.height<0 || rightHand.width<0)
					{
						rightHand.x = 0;
						rightHand.y = 0;
						rightHand.height = 10;
						rightHand.width = 10;
					}
					if(leftHand.x<0 || leftHand.y<0 || leftHand.height<0 || leftHand.width<0)
					{
						leftHand.x = 0;
						leftHand.y = 0;
						leftHand.height = 10;
						leftHand.width = 10;
					}

					//vPosture.push_back(posture); 

					CString imgFileName_left;
					CString imgFileName_right;
					imgFileName_left.Format(out_imgFileFolder + "\\%04d_L.bmp",i);
					imgFileName_right.Format(out_imgFileFolder + "\\%04d_R.bmp",i);

					
					cvSaveImage(imgFileName_left, posture.leftHandImg);
					cvSaveImage(imgFileName_right, posture.rightHandImg);
					/*double hog_color[DES_FEA_NUM];
					handSegmentVideo.getHogFeature(posture.leftHandImg, posture.rightHandImg,hog_color);

					for (int d=0; d<DES_FEA_NUM; d++)
					{
					outfile<<hog_color[d]<<" ";
					}
					outfile<<endl;*/
					//////////////////////////////////////////////////////////////////////////
					//Tskp
					//Vector<float> skP;
					//for (int sk=3; sk<13; sk+=2)
					//{
					//	for (int sk2=sk+2; sk2<13; sk2+=2)
					//	{
					//		float temp = pow((myReadVideo.vSkeletonData[i]._3dPoint[sk].x - myReadVideo.vSkeletonData[i]._3dPoint[sk2].x),2)
					//			+ pow((myReadVideo.vSkeletonData[i]._3dPoint[sk].y - myReadVideo.vSkeletonData[i]._3dPoint[sk2].y),2)
					//			+ pow((myReadVideo.vSkeletonData[i]._3dPoint[sk].z - myReadVideo.vSkeletonData[i]._3dPoint[sk2].z),2);
					//		skP.push_back(temp);
					//	}
					//}

					//float maxSKP = 0.0;
					//for (int k=0; k<skP.size(); k++)
					//{
					//	if (maxSKP < skP[k])
					//	{
					//		maxSKP = skP[k];
					//	}
					//}
					//for (int k=0; k<skP.size(); k++)
					//{
					//	skP[k] /=maxSKP;
					//	outfile<<skP[k]<<" ";
					//}
					//outfile<<endl;
					//////////////////////////////////////////////////////////////////////////
					//depth image
					totalFrameNum++;
					//vector<double> hog_hist;
					//vector<double> hog_hist_reduced;
					
					Mat depthImage = myReadVideo.retrieveColorDepth(myReadVideo.vDepthData[i]);
					imgFileName_left.Format(out_DepthImgFileFolder + "\\%04d_L.bmp",i);
					imgFileName_right.Format(out_DepthImgFileFolder + "\\%04d_R.bmp",i);


						//output the images
// 					cvRectangle(myReadVideo.vColorData[i],cvPoint(leftHand.x,leftHand.y),cvPoint(leftHand.x+leftHand.width, leftHand.y+leftHand.height)
// 						,cvScalar(0,255,0),2,8,0);
// 					cvRectangle(myReadVideo.vColorData[i],cvPoint(rightHand.x,rightHand.y),cvPoint(rightHand.x+rightHand.width, rightHand.y+rightHand.height)
// 						,cvScalar(0,255,0),2,8,0);
// 					IplImage* black = cvCreateImage(cvSize(640,480),8,3);
					
// 					for (int j=2; j<=11; j++)
// 					{
// 						CvPoint p1;
// 						p1.x = myReadVideo.vSkeletonData[i]._2dPoint[j].x;
// 						p1.y = myReadVideo.vSkeletonData[i]._2dPoint[j].y;
// 						cvCircle(myReadVideo.vColorData[i],p1,2,cvScalar(255,0,0),2,8,0);
// 						if (j!=7 && j!=3 && j!=11)
// 						{
// 							CvPoint p2;
// 							p2.x = myReadVideo.vSkeletonData[i]._2dPoint[j+1].x;
// 							p2.y = myReadVideo.vSkeletonData[i]._2dPoint[j+1].y;
// 							cvLine(myReadVideo.vColorData[i],p1,p2,cvScalar(0,0,255),2,8,0);
// 						}
// 						if (j==2)
// 						{
// 							CvPoint p2;
// 							p2.x = myReadVideo.vSkeletonData[i]._2dPoint[j+2].x;
// 							p2.y = myReadVideo.vSkeletonData[i]._2dPoint[j+2].y;
// 							CvPoint p3;
// 							p3.x = myReadVideo.vSkeletonData[i]._2dPoint[j+6].x;
// 							p3.y = myReadVideo.vSkeletonData[i]._2dPoint[j+6].y;
// 							cvLine(myReadVideo.vColorData[i],p1,p2,cvScalar(0,0,255),2,8,0);
// 							cvLine(myReadVideo.vColorData[i],p1,p3,cvScalar(0,0,255),2,8,0);
// 						}
// 
// 					}
// 					for (int j=3; j<13; j+=2)
// 					{
// 						CvPoint p1;
// 						p1.x = myReadVideo.vSkeletonData[i]._2dPoint[j].x;
// 						p1.y = myReadVideo.vSkeletonData[i]._2dPoint[j].y;
// 						for (int k=j+2; k<13; k+=2)
// 						{
// 							CvPoint p2;
// 							p2.x = myReadVideo.vSkeletonData[i]._2dPoint[k].x;
// 							p2.y = myReadVideo.vSkeletonData[i]._2dPoint[k].y;
// 							cvLine(myReadVideo.vColorData[i],p1,p2,cvScalar(0,255,0),1,8,0);
// 						}
// 					}
// 					cvSaveImage(imgFileName_left, myReadVideo.vColorData[i]);
						//Depth images are saved in this function

					IplImage* leftHandImg = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,3);
					IplImage* rightHandImg = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,3);
// 					IplImage* image = &(IplImage)depthImage;
// 					cvSetImageROI(image,leftHand);
// 					cvResize(image, leftHandImg, 1);
// 					cvResetImageROI(image);
// 					IplImage* leftHandImg_ = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,1);
// 					cvCvtColor(leftHandImg,leftHandImg_,CV_BGR2GRAY);
// 					cvSaveImage(imgFileName_left,leftHandImg_);

					
					IplImage* image = &(IplImage)depthImage;
					cvSetImageROI(image,leftHand);
					IplImage* leftHandImg_ = cvCreateImage(cvSize(leftHand.width, leftHand.height),8,1);
					cvCvtColor(image,leftHandImg_,CV_BGR2GRAY);
					cvSaveImage(imgFileName_left,leftHandImg_);
					cvResetImageROI(image);

// 					cvSetImageROI(image,rightHand);
// 					cvResize(image, rightHandImg, 1);
// 					cvResetImageROI(image);
// 					IplImage* rightHandImg_ = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,1);
// 					cvCvtColor(rightHandImg,rightHandImg_,CV_BGR2GRAY);
// 					cvSaveImage(imgFileName_right,rightHandImg_);

					cvSetImageROI(image,rightHand);
					IplImage* rightHandImg_ = cvCreateImage(cvSize(rightHand.width, rightHand.height),8,1);
					cvCvtColor(image,rightHandImg_,CV_BGR2GRAY);
					cvSaveImage(imgFileName_right,rightHandImg_);
					cvResetImageROI(image);

					cvReleaseImage(&leftHandImg);
					cvReleaseImage(&leftHandImg_);
					cvReleaseImage(&rightHandImg);
					cvReleaseImage(&rightHandImg_);

// 					bool handRetriveFlag = HandFeatureRetrieve(&(IplImage)depthImage, 
// 						myReadVideo.vSkeletonData[i], hog_hist,imgFileName_left,imgFileName_right,leftHand,rightHand);
// 					featureReduce(hog_hist, hog_hist_reduced);
					finish_feature = clock();
					timeForFeatureExtraction = (float)(finish_feature - start_feature);

					//
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[3].x<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[3].y<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[3].z<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[5].x<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[5].y<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[5].z<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[7].x<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[7].y<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[7].z<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[9].x<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[9].y<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[9].z<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[11].x<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[11].y<<" ";
					outfile<<myReadVideo.vSkeletonData[i]._3dPoint[11].z<<" ";
					outfile<<endl;
// 					for (int d=0; d<hog_hist_reduced.size(); d++)
// 					{
// 						outfile<<(hog_hist_reduced[d]-(-0.0295))/(0.0149 - (-0.0295))<<" ";
// 					}
// 					outfile<<endl;

					//输出用于训练PCA的data的代码
					//if (i<6)
					//{
					//	outfileAll.open("..\\output\\HOGofAll.txt", ios::app | ios::out);
					//	for (int d=0; d<hog_hist_reduced.size(); d++)
					//	{
					//		outfileAll<<hog_hist_reduced[d]<<'\t';
					//	}
					//	outfileAll<<endl;
					//	outfileAll.close();
					//}

					//Show images
					//cvShowImage(videoName,&(IplImage)depthImage);
					////cvShowImage(videoName,myReadVideo.vColorData[i]);
					//cvWaitKey(0);
				}

					
			}

				//For output the C feature
// 			for(int i=0; i<vPosture.size(); i++)
// 			{
// 				Posture posture = vPosture[i];
// 				if(posture.leftHandImg == NULL || posture.rightHandImg == NULL)
// 				{
// 					for(int j=i-1; j>=0; j--)
// 					{
// 						Posture tmpPosture = vPosture[j];
// 						if(posture.leftHandImg == NULL && tmpPosture.leftHandImg != NULL)
// 						{
// 							posture.leftHandImg = cvCreateImage(cvGetSize(tmpPosture.leftHandImg),tmpPosture.leftHandImg->depth,tmpPosture.leftHandImg->nChannels);
// 							cvCopy(tmpPosture.leftHandImg,posture.leftHandImg);
// 						}
// 						if(posture.rightHandImg == NULL && tmpPosture.rightHandImg != NULL)
// 						{
// 							posture.rightHandImg = cvCreateImage(cvGetSize(tmpPosture.rightHandImg),tmpPosture.rightHandImg->depth,tmpPosture.rightHandImg->nChannels);
// 							cvCopy(tmpPosture.rightHandImg,posture.rightHandImg);
// 						}
// 						if(posture.rightHandImg != NULL && posture.leftHandImg != NULL)
// 						{
// 							break;
// 						}
// 					}
// 				}
// 
// 				double hog_color[DES_FEA_NUM];
// 				handSegmentVideo.getHogFeature(posture.leftHandImg, posture.rightHandImg,hog_color);

// 				for (int d=0; d<DES_FEA_NUM; d++)
// 				{
// 					outfile<<hog_color[d]<<" ";
// 				}
// 				outfile<<endl;
//			}

				//Release the C resource
// 			for (int i=0; i<vPosture.size(); i++)
// 			{
// 				cvReleaseImage(&vPosture[i].leftHandImg);
// 				cvReleaseImage(&vPosture[i].rightHandImg);
// 			}
// 			vPosture.clear();

			outfile.close();
			//cvDestroyWindow(videoName);
			myReadVideo.release();
			videoTotalNum += 1;
			cout<<videoName<<" total: "<<frameSize<<", used: "<<framSize_Real<<endl;
			cout<<endl;
		}
	}

	//finish = clock();
	timeForReadvideo = timeForReadvideo / (CLOCKS_PER_SEC);  //All the vedios
	timeForFeatureExtraction = timeForFeatureExtraction / (CLOCKS_PER_SEC * totalFrameNum);  //Each frame.
	cout<<"Time for reading video: "<<timeForReadvideo<<endl;
	cout<<"Time for extracting each frame: "<<timeForFeatureExtraction<<endl;

#endif

#ifdef mergeFeature

	CFileFind txtFileFind_P;
	CString txtfileFolderName_P;
	BOOL txtFindFlag_P;
	CFileFind txtFileFind_T;
	CString txtfileFolderName_T;
	BOOL txtFindFlag_T;

	ofstream mergeOutfile;
	for (int myGroupID = 0; myGroupID<GroupNum; myGroupID++)
	{
		int testFolderID = groupID[myGroupID];
		int testFolderID_sub = subgroupID[myGroupID];

		cout<<"Processing: P0"<<testFolderID<<"_"<<testFolderID_sub<<"..."<<endl;
		CString dstFileTrain;
		CString dstFileTest;
		dstFileTrain.Format(dir + "train_%02d_%d",testFolderID, testFolderID_sub);
		_mkdir(dstFileTrain);
		dstFileTest.Format(dir + "test_%02d_%d",testFolderID, testFolderID_sub);
		_mkdir(dstFileTest);

		for (int myGroupID_t=0; myGroupID_t<GroupNum; myGroupID_t++)
		{
			int folderID = groupID[myGroupID_t];
			int folderID_sub = subgroupID[myGroupID_t];
			txtfileFolderName_P.Format(dir + "posture\\P%02d_%d\\*.txt",folderID, folderID_sub);
			txtfileFolderName_T.Format(dir + "trajectory\\P%02d_%d\\*.txt",folderID, folderID_sub);
			txtFindFlag_P = TRUE;
			txtFindFlag_P = txtFileFind_P.FindFile(txtfileFolderName_P);
			txtFindFlag_T = TRUE;
			txtFindFlag_T = txtFileFind_T.FindFile(txtfileFolderName_T);
			while(txtFindFlag_P && txtFindFlag_T)
			{ 
				txtFindFlag_P = txtFileFind_P.FindNextFile();
				txtFindFlag_T = txtFileFind_T.FindNextFile();
				CString txtFileName_P = txtFileFind_P.GetFilePath();
				CString txtFileName_T = txtFileFind_T.GetFilePath();

				cout<<"Processing: P0"<<testFolderID<<"_"<<testFolderID_sub<<" P0"<<folderID<<"_"
					<<folderID_sub<<" "<<txtFileName_P.Right(8).Left(4)<<"..."<<endl;

				CString mergeOutfileName_train;
				if (folderID == testFolderID && folderID_sub == testFolderID_sub)
				{
					mergeOutfileName_train.Format(dir + "test_%02d_%d\\w"+txtFileName_P.Right(8).Left(4)+".txt"
						,testFolderID, testFolderID_sub);  //8,4
				}
				else
				{
					//continue;
					mergeOutfileName_train.Format(dir + "train_%02d_%d\\w"+txtFileName_P.Right(8).Left(4)+".txt"
						,testFolderID, testFolderID_sub);
				}

				mergeOutfile.open(mergeOutfileName_train, ios::app | ios::out);

				int frame,feaNums;
				float featureNum;
				fstream f1(txtFileName_P,ios::in);
				f1 >> frame >> feaNums;

				int frame_T,feaNums_T;
				float featureNum_T;
				fstream f2(txtFileName_T,ios::in);
				f2 >> frame_T >> feaNums_T;

				mergeOutfile << frame << " " << feaNums+feaNums_T << endl;

				int i, j;
				for(i=0; i<frame; i++)
				{
					for(j=0; j<feaNums_T; j++)
					{
						f2 >> featureNum_T;
						mergeOutfile << featureNum_T << " ";
					}

					for(j=0; j<feaNums; j++)
					{
						f1 >> featureNum;
						mergeOutfile << featureNum << " ";
					}
					mergeOutfile << endl;
				}
				f1.close();
				mergeOutfile.close();
				//////////////////////////////////////////////////////////////////////////
#ifdef CRFOutput

				CString mergeOutfileName_train_csv;
				CString mergeOutfileName_train_label_csv;
				if (folderID == testFolderID)
				{
					mergeOutfileName_train_csv.Format(dir + "dataTest_%d.csv",testFolderID);  //8,4
					mergeOutfileName_train_label_csv.Format(dir + "labelsTest_%d.csv",testFolderID);
				}
				else
				{
					mergeOutfileName_train_csv.Format(dir + "dataTrain_%d.csv",testFolderID);
					mergeOutfileName_train_label_csv.Format(dir + "labelsTrain_%d.csv",testFolderID);
				}
				ofstream mergeOutfile_CSV;
				ofstream mergeOutfile_label_CSV;
				mergeOutfile_CSV.open(mergeOutfileName_train_csv, ios::app | ios::out);
				mergeOutfile_label_CSV.open(mergeOutfileName_train_label_csv, ios::app | ios::out);
				int frame_csv,feaNums_csv;
				float featureNum_csv;
				fstream f1_csv(txtFileName_P,ios::in);
				f1_csv >> frame_csv >> feaNums_csv;

				int frame_T_csv,feaNums_T_csv;
				float featureNum_T_csv;
				fstream f2_csv(txtFileName_T,ios::in);
				f2_csv >> frame_T_csv >> feaNums_T_csv;



				float** tempdata;
				tempdata = new float*[frame_csv];
				for (int i=0; i<frame_csv; i++)
				{
					tempdata[i] = new float[feaNums_csv+feaNums_T_csv];
				}
				for(int i=0; i<frame_csv; i++)
				{
					for(j=0; j<feaNums_T_csv; j++)
					{
						f2_csv >> featureNum_T_csv;
						tempdata[i][j] = featureNum_T_csv;
					}

					for(j=0; j<feaNums_csv; j++)
					{
						f1_csv >> featureNum_csv;
						tempdata[i][j+feaNums_T_csv] = featureNum_csv;
					}
				}

				mergeOutfile_CSV << feaNums_csv+feaNums_T_csv << "," << frame_csv << endl;
				mergeOutfile_label_CSV << 1 << "," << feaNums_csv+feaNums_T_csv << endl;
				for (int i=0; i<feaNums_csv+feaNums_T_csv; i++)
				{
					for (int j=0; j<frame_csv; j++)
					{
						mergeOutfile_CSV<<tempdata[j][i]<<",";
					}
					mergeOutfile_CSV<<endl;
					mergeOutfile_label_CSV<<txtFileName_P.Right(8).Left(4)<<",";
				}
				mergeOutfile_label_CSV<<endl;
				//mergeOutfile_label_CSV<<txtFileName_P.Right(8).Left(4)<<endl;

				for (int i=0; i<frame_csv; i++)
				{
					delete[] tempdata[i];
				}
				delete tempdata;

				f1_csv.close();
				f2_csv.close();
				mergeOutfile_CSV.close();
#endif

			}
		}
	}


#endif

#ifdef OneM_mergeFeature

	CFileFind txtFileFind_P;
	CString txtfileFolderName_P;
	BOOL txtFindFlag_P;
	CFileFind txtFileFind_T;
	CString txtfileFolderName_T;
	BOOL txtFindFlag_T;

	ofstream mergeOutfile;
	for (int myGroupID_t=0; myGroupID_t<GroupNum; myGroupID_t++)
	{
		int folderID = groupID[myGroupID_t];
		int folderID_sub = subgroupID[myGroupID_t];
		cout<<"Processing: P0"<<folderID<<"_"<<folderID_sub<<"..."<<endl;

			//Create the folders
		CString dstFile;
		dstFile.Format(dir + "P%02d_%d",folderID, folderID_sub);
		_mkdir(dstFile);

			//Merge the posture and trajectory data
		txtfileFolderName_P.Format(dir + "posture\\P%02d_%d\\*.txt",folderID, folderID_sub);
		txtfileFolderName_T.Format(dir + "trajectory\\P%02d_%d\\*.txt",folderID, folderID_sub);
		txtFindFlag_P = TRUE;
		txtFindFlag_P = txtFileFind_P.FindFile(txtfileFolderName_P);
		txtFindFlag_T = TRUE;
		txtFindFlag_T = txtFileFind_T.FindFile(txtfileFolderName_T);
		while(txtFindFlag_P && txtFindFlag_T)
		{ 
			txtFindFlag_P = txtFileFind_P.FindNextFile();
			txtFindFlag_T = txtFileFind_T.FindNextFile();
			CString txtFileName_P = txtFileFind_P.GetFilePath();
			CString txtFileName_T = txtFileFind_T.GetFilePath();

			cout<<"Processing: P0"<<folderID<<"_"
				<<folderID_sub<<" "<<txtFileName_P.Right(8).Left(4)<<"..."<<endl;

			CString mergeOutfileName_train;
			mergeOutfileName_train.Format(dir + "P%02d_%d\\P%02d_%d_"+txtFileName_P.Right(8).Left(4)+".txt"
				,folderID, folderID_sub,folderID, folderID_sub);  //8,4

			mergeOutfile.open(mergeOutfileName_train, ios::app | ios::out);

			int frame,feaNums;
			float featureNum;
			fstream f1(txtFileName_P,ios::in);
			f1 >> frame >> feaNums;

			int frame_T,feaNums_T;
			float featureNum_T;
			fstream f2(txtFileName_T,ios::in);
			f2 >> frame_T >> feaNums_T;

			mergeOutfile << frame << " " << feaNums+feaNums_T << endl;

			int i, j;
			for(i=0; i<frame; i++)
			{
				for(j=0; j<feaNums_T; j++)
				{
					f2 >> featureNum_T;
					mergeOutfile << featureNum_T << " ";
				}

				for(j=0; j<feaNums; j++)
				{
					f1 >> featureNum;
					mergeOutfile << featureNum << " ";
				}
				mergeOutfile << endl;
			}
			f1.close();
			mergeOutfile.close();

		}
	}
#endif

#ifdef OutputReorganize
	CFileFind txtFileFind;
	CString txtfileFolderName;
	BOOL txtFindFlag;
	ofstream mergeOutfile;
	//CString dir = "..\\output\\dimension_51C_oldSDK\\";

	for (int myGroupID = 1; myGroupID<GroupNum; myGroupID++)
	{
		int testFolderID = groupID[myGroupID];
		int testFolderID_sub = subgroupID[myGroupID];
		cout<<"Processing: "<<testFolderID<<"_"<<testFolderID_sub<<"..."<<endl;
		CString dstFileTrain;
		CString dstFileTest;
		dstFileTrain.Format(dir + "train_%02d_%d",testFolderID, testFolderID_sub);
		_mkdir(dstFileTrain);
		dstFileTest.Format(dir + "test_%02d_%d",testFolderID, testFolderID_sub);
		_mkdir(dstFileTest);

		for (int myGroupID_t=0; myGroupID_t<GroupNum; myGroupID_t++)
		{
			
			int folderID = groupID[myGroupID_t];
			int folderID_sub = subgroupID[myGroupID_t];
			
			txtfileFolderName.Format(dir + "P%02d_%d\\*.txt",folderID, folderID_sub);
			txtFindFlag = TRUE;
			txtFindFlag = txtFileFind.FindFile(txtfileFolderName);
			while(txtFindFlag)
			{ 
				txtFindFlag = txtFileFind.FindNextFile();
				CString txtFileName = txtFileFind.GetFilePath();
	
				cout<<"Processing: P0"<<testFolderID<<"_"<<testFolderID_sub<<" P0"<<folderID<<"_"
					<<folderID_sub<<" "<<txtFileName.Right(8).Left(4)<<"..."<<endl;

				CString mergeOutfileName_train;
				if (folderID == testFolderID && folderID_sub == testFolderID_sub)
				{
					mergeOutfileName_train.Format(dir + "test_%02d_%d\\w"+txtFileName.Right(8).Left(4)+".txt"
						,testFolderID, testFolderID_sub);  //8,4
				}
				else
				{
					mergeOutfileName_train.Format(dir + "train_%02d_%d\\w"+txtFileName.Right(8).Left(4)+".txt"
						,testFolderID, testFolderID_sub);
				}

				mergeOutfile.open(mergeOutfileName_train, ios::app | ios::out);

				int frame,feaNums;
				float featureNum;
				fstream f1(txtFileName,ios::in);
				f1 >> frame >> feaNums;
				mergeOutfile << frame << " " << feaNums << endl;
				int i, j;
				for(i=0; i<frame; i++)
				{
					for(j=0; j<feaNums; j++)
					{
						f1 >> featureNum;
						mergeOutfile << featureNum << " ";
					}
					mergeOutfile << endl;
				}
				f1.close();
				mergeOutfile.close();
			}
		}
	}



#endif

#ifdef OneM_OutputReorganize
	CFileFind txtFileFind;
	CString txtfileFolderName;
	BOOL txtFindFlag;
	ofstream mergeOutfile;

		//Prepare for training data
	for (int myGroupID = 0; myGroupID<8; myGroupID++)
	{
		int folderID = groupID[myGroupID];
		int folderID_sub = subgroupID[myGroupID];
		cout<<"Processing training data: P"<<folderID<<"_"<<folderID_sub<<"..."<<endl;
		CString dstFileTrain;
		dstFileTrain.Format(dir + "trainData");
		_mkdir(dstFileTrain);

		txtfileFolderName.Format(dir + "P%02d_%d\\*.txt",folderID, folderID_sub);
		txtFindFlag = TRUE;
		txtFindFlag = txtFileFind.FindFile(txtfileFolderName);

		while(txtFindFlag)
		{ 
			txtFindFlag = txtFileFind.FindNextFile();
			CString txtFileName = txtFileFind.GetFilePath();

			cout<<"Processing training data: P0"<<folderID<<"_"
				<<folderID_sub<<" "<<txtFileName.Right(8).Left(4)<<"..."<<endl;

			CString mergeOutfileName_train;
			mergeOutfileName_train.Format(dir + "trainData\\w"+txtFileName.Right(8).Left(4)+".txt");
			mergeOutfile.open(mergeOutfileName_train, ios::app | ios::out);

			int frame,feaNums;
			float featureNum;
			fstream f1(txtFileName,ios::in);
			f1 >> frame >> feaNums;
			mergeOutfile << frame << " " << feaNums << endl;
			int i, j;
			for(i=0; i<frame; i++)
			{
				for(j=0; j<feaNums; j++)
				{
					f1 >> featureNum;
					mergeOutfile << featureNum << " ";
				}
				mergeOutfile << endl;
			}
			f1.close();
			mergeOutfile.close();
		}
	}

		//Prepare for test data
	for (int myGroupID=8; myGroupID<GroupNum; myGroupID++)
	{
		int folderID = groupID[myGroupID];
		int folderID_sub = subgroupID[myGroupID];

		CString dstFileTest;
		dstFileTest.Format(dir + "test_%02d_%d",folderID, folderID_sub);
		_mkdir(dstFileTest);


		txtfileFolderName.Format(dir + "P%02d_%d\\*.txt",folderID, folderID_sub);
		txtFindFlag = TRUE;
		txtFindFlag = txtFileFind.FindFile(txtfileFolderName);
		while(txtFindFlag)
		{ 
			txtFindFlag = txtFileFind.FindNextFile();
			CString txtFileName = txtFileFind.GetFilePath();

			cout<<"Processing test data: P0"<<folderID<<"_"
				<<folderID_sub<<" "<<txtFileName.Right(8).Left(4)<<"..."<<endl;

			CString mergeOutfileName_train;
			
			mergeOutfileName_train.Format(dir + "test_%02d_%d\\w"+txtFileName.Right(8).Left(4)+".txt"
					,folderID, folderID_sub);  //8,4
			mergeOutfile.open(mergeOutfileName_train, ios::app | ios::out);

			int frame,feaNums;
			float featureNum;
			fstream f1(txtFileName,ios::in);
			f1 >> frame >> feaNums;
			mergeOutfile << frame << " " << feaNums << endl;
			int i, j;
			for(i=0; i<frame; i++)
			{
				for(j=0; j<feaNums; j++)
				{
					f1 >> featureNum;
					mergeOutfile << featureNum << " ";
				}
				mergeOutfile << endl;
			}
			f1.close();
			mergeOutfile.close();
		}
	}
#endif

//////////////////////////////////////////////////////////////////////////
//Training and test with HMM
//////////////////////////////////////////////////////////////////////////
		// Record the total state number
	int totalStates = 0;

		//Settings for recording the cost time
	clock_t start_all, finish_all;
	start_all = clock();
	CString timeCostOutput;
	timeCostOutput.Format(dir + "Time_cost.txt");
	ofstream timeCostFile;
	timeCostFile.open(timeCostOutput, ios::app | ios::out);

#ifdef TRAIN
	char sourcefile[500], trainfile[500], resultfile[500], word[600];
	//char dir[600];
	FILE *fp1, *fp2;
	int count = 0;
	CHMM* m_pDhmm; 
	
	//int m_nHMM_Samples = folderEnd - folderStart -1;
	int m_nHMM_Samples = GroupNum -1;
	int m_nHMM_MixT = 3;
	int m_nHMM_MixS = 3;
	int m_nHMM_dimension = DIMENSION;

	m_pDhmm = new CHMM;
	m_pDhmm->m_nMaxStateSize = m_nHMM_MixT;
	m_pDhmm->m_nDimension = m_nHMM_dimension;
	m_pDhmm->m_nMixS = m_nHMM_MixS;
	m_pDhmm->m_bFlagTrain = TRUE;    //training.

	
	
	for (int myGroupID = 0; myGroupID < GroupNum; myGroupID++)
	{
		
		int folderName = groupID[myGroupID];
		int folderName_sub = subgroupID[myGroupID];
		if ((fp1 = fopen(signListName, "r")) == NULL) {
			AfxMessageBox("Invalid source file.");
			return 0;
		}
		fscanf(fp1, "%d\n", &count);
		m_pDhmm->m_nTotalHmmWord = 0;

		CString resultfileFolder;
		resultfileFolder.Format(dir + "hmmcode_%02d_%d", folderName, folderName_sub);
		_mkdir(resultfileFolder);

		for (int i=0; i<count; i++) 
		{
			fscanf(fp1, "%s", word);
			cout<<"Generating P"<<folderName<<"_"<<folderName_sub<<": "<<word<<".hmm "<<endl;

			sprintf(trainfile, "%s%s%02d%s%d%s%s%s", dir, "train_", folderName, "_", folderName_sub, "\\", word, ".txt");
			sprintf(resultfile, "%s%s%02d%s%d%s%s%s", dir, "hmmcode_", folderName, "_", folderName_sub, "\\", word, ".hmm");

			if ((fp2 = fopen(trainfile, "r")) == NULL) 
			{
				continue;
			}
			fclose(fp2);

			m_pDhmm->m_nTotalHmmWord += 1;

			//如果数据归一，只修改这里即可。
			//如果数据没有归一，还要修改HMM.cpp中Getdata的内容，写入归一信息.
			int stateNum= m_pDhmm->DHMM(trainfile, resultfile, m_nHMM_Samples, m_nHMM_MixT, m_nHMM_MixS, TRUE);  //数据已经归一
			//m_pDhmm->DHMM(trainfile, resultfile, m_nHMM_Samples, m_nHMM_MixT, m_nHMM_MixS, FALSE);  // FALSE 没有归一

			totalStates += stateNum;
		}

		fclose(fp1); 

		cout<<"Training finished!"<<endl;
		CString binaryOutFileName;
		binaryOutFileName.Format(dir + "HmmData_%02d_%d.dat", folderName, folderName_sub);
		m_pDhmm->ConvertSourceHMMToBinary(resultfileFolder, binaryOutFileName, signListName);
		cout<<"Binary file P"<<folderName<<" generation finished!"<<endl;
	}

	
#endif

#ifdef OneM_TRAIN
	char sourcefile[500], trainfile[500], resultfile[500], word[600];
	//char dir[600];
	FILE *fp1, *fp2;
	int count = 0;
	CHMM* m_pDhmm; 

	int m_nHMM_Samples = trainNum;
	int m_nHMM_MixT = 3;
	int m_nHMM_MixS = 3;
	int m_nHMM_dimension = DIMENSION;

	m_pDhmm = new CHMM;
	m_pDhmm->m_nMaxStateSize = m_nHMM_MixT;
	m_pDhmm->m_nDimension = m_nHMM_dimension;
	m_pDhmm->m_nMixS = m_nHMM_MixS;
	m_pDhmm->m_bFlagTrain = TRUE;    //training.

	if ((fp1 = fopen(signListName, "r")) == NULL) {
		AfxMessageBox("Invalid source file.");
		return 0;
	}
	fscanf(fp1, "%d\n", &count);
	m_pDhmm->m_nTotalHmmWord = 0;

	CString resultfileFolder;
	resultfileFolder.Format(dir + "hmmcode");
	_mkdir(resultfileFolder);

	for (int i=0; i<count; i++) 
	{
		fscanf(fp1, "%s", word);
		cout<<"Generating "<<word<<".hmm "<<endl;

		sprintf(trainfile, "%s%s%s%s%s", dir, "trainData", "\\", word, ".txt");
		sprintf(resultfile, "%s%s%s%s%s", dir, "hmmcode", "\\", word, ".hmm");

		if ((fp2 = fopen(trainfile, "r")) == NULL) 
		{
			continue;
		}
		fclose(fp2);

		m_pDhmm->m_nTotalHmmWord += 1;

		int stateNum= m_pDhmm->DHMM(trainfile, resultfile, m_nHMM_Samples, m_nHMM_MixT, m_nHMM_MixS, TRUE);  //数据已经归一
		//m_pDhmm->DHMM(trainfile, resultfile, m_nHMM_Samples, m_nHMM_MixT, m_nHMM_MixS, FALSE);  // FALSE 没有归一

		totalStates += stateNum;
	}

	fclose(fp1); 

	cout<<"Training finished!"<<endl;
	CString binaryOutFileName;
	binaryOutFileName.Format(dir + "HmmData.dat");
	m_pDhmm->ConvertSourceHMMToBinary(resultfileFolder, binaryOutFileName, signListName);
	cout<<"Binary file generation finished!"<<endl;

#endif

#ifdef TEST
	CHMM* m_pDhmm_test;
	CRecognition *m_pRecog;
	m_pDhmm_test = new CHMM;
	
	m_pRecog = new CRecognition;
	
// 	CString TestFile;
	clock_t start, finish;
	start = clock();
	for (int myGroupID = 0; myGroupID<GroupNum; myGroupID++)
	{
		int testID = groupID[myGroupID];
		int folderName_sub = subgroupID[myGroupID];
		CString hmmDataName;
		hmmDataName.Format(dir + "HmmData_%02d_%d.dat", testID, folderName_sub);
		m_pDhmm_test->Init(hmmDataName);
		m_pRecog->GetHmmModel(m_pDhmm_test);
		
		m_pRecog->BatchTest(signListName, dir, testID, folderName_sub);
		
	}
	finish = clock();
	float duration = (float)(finish - start) ;/// (CLOCKS_PER_SEC*m_pDhmm_test->m_nTotalHmmWord);
	cout<<"Test duration for all the signs: "<<duration<<endl;
	
#endif

#ifdef OneM_TEST
	CHMM* m_pDhmm_test;
	CRecognition *m_pRecog;
	m_pDhmm_test = new CHMM;

	m_pRecog = new CRecognition;

	// 	CString TestFile;
	clock_t start, finish;
	start = clock();
	CString hmmDataName;
	hmmDataName.Format(dir + "HmmData.dat");
	m_pDhmm_test->Init(hmmDataName);
	m_pRecog->GetHmmModel(m_pDhmm_test);

	for (int myGroupID = 8; myGroupID<GroupNum; myGroupID++)
	{
		int testID = groupID[myGroupID];
		int folderName_sub = subgroupID[myGroupID];
		m_pRecog->BatchTest(signListName, dir, testID, folderName_sub);

	}
	finish = clock();
	float duration = (float)(finish - start) ;/// (CLOCKS_PER_SEC*m_pDhmm_test->m_nTotalHmmWord);
	cout<<"Test duration for all the signs: "<<duration<<endl;
	timeCostFile<<"Test: "<<duration/(MAXWORDNUM*(GroupNum-trainNum))<<" ms / sign"<<endl;
#endif

	finish_all = clock();
	float timeForprocessing = (float)(finish_all - start_all);
	cout<<"Time cost: "<<timeForprocessing<<endl;

#ifdef OneM_TEST
	timeCostFile<<"Train: "<<(timeForprocessing-duration)/(MAXWORDNUM)<<" ms / sign (each has "<<trainNum<<" samples)"<<endl;
	cout<<"Total states: "<<totalStates<<endl;  //Train should be useful
	timeCostFile<<"Total states: "<<totalStates<<endl;
	timeCostFile<<"Class Number: "<<MAXWORDNUM<<endl;
	timeCostFile<<"Train samples: "<<trainNum<<endl;
	timeCostFile<<"Test samples: "<<GroupNum-trainNum<<endl;
	timeCostFile.close();
#endif

	cout<<"Done!"<<endl;
	
	getchar();
	return 1;
}

bool GetHOGHistogram_Patch(IplImage *img,vector<double> &hog_hist)//取得图像img的HOG特征向量
{
	//HOGDescriptor *hog=new HOGDescriptor(cvSize(SIZE,SIZE),cvSize(8,8),cvSize(4,4),cvSize(4,4),9);
	//HOGDescriptor *hog=new HOGDescriptor(cvSize(PostureSIZE,PostureSIZE),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
	HOGDescriptor *hog=new HOGDescriptor(cvSize(PostureSIZE,PostureSIZE),cvSize(32,32),cvSize(16,16),cvSize(16,16),9);
	/////////////////////window大小为64*64，block大小为8*8，block步长为4*4，cell大小为4*4
	Mat handMat(img);

	vector<float> *descriptors = new std::vector<float>();

	hog->compute(handMat, *descriptors,Size(0,0), Size(0,0));

	////////////////////window步长为0*0
	double total=0;
	int i;
	for(i=0;i<descriptors->size();i++)
		total+=abs((*descriptors)[i]);
	//	total=sqrt(total);
	for(i=0;i<descriptors->size();i++)
		hog_hist.push_back((*descriptors)[i]/total);
	return true; 
}

// bool GetHOGHistogram_OriDepth(Mat handMat,vector<double> &hog_hist)//取得图像img的HOG特征向量
// {
// 	//HOGDescriptor *hog=new HOGDescriptor(cvSize(SIZE,SIZE),cvSize(8,8),cvSize(4,4),cvSize(4,4),9);
// 	//HOGDescriptor *hog=new HOGDescriptor(cvSize(PostureSIZE,PostureSIZE),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
// 	HOGDescriptor *hog=new HOGDescriptor(cvSize(PostureSIZE,PostureSIZE),cvSize(32,32),cvSize(16,16),cvSize(16,16),9);
// 	/////////////////////window大小为64*64，block大小为8*8，block步长为4*4，cell大小为4*4
// 	//Mat handMat(img);
// 
// 	vector<float> *descriptors = new std::vector<float>();
// 
// 	hog->compute(handMat, *descriptors,Size(0,0), Size(0,0));
// 
// 	////////////////////window步长为0*0
// 	double total=0;
// 	int i;
// 	for(i=0;i<descriptors->size();i++)
// 		total+=abs((*descriptors)[i]);
// 	//	total=sqrt(total);
// 	for(i=0;i<descriptors->size();i++)
// 		hog_hist.push_back((*descriptors)[i]/total);
// 	return true; 
// }

bool HandFeatureRetrieve(IplImage* image,SLR_ST_Skeleton skeletonData,vector<double> &hog_hist, CString leftHandFileName, CString rightHandFileName
	,CvRect leftHand, CvRect rightHand)
{
	bool showImage = FALSE;
	IplImage* leftHandImg = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,3);
	IplImage* rightHandImg = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,3);
	vector<double> hog_hist_left;
	vector<double> hog_hist_right;

	for (int j=2; j<=11; j++)
	{
		CvPoint p1;
		p1.x = skeletonData._2dPoint[j].x;
		p1.y = skeletonData._2dPoint[j].y;
		if (showImage)
		{
			cvCircle(image,p1,2,cvScalar(225,0,0),2,8,0);
			if (j!=7 && j!=3 && j!=11)
			{
				CvPoint p2;
				p2.x = skeletonData._2dPoint[j+1].x;
				p2.y = skeletonData._2dPoint[j+1].y;
				cvLine(image,p1,p2,cvScalar(0,0,225),2,8,0);
			}
			if (j==2)
			{
				CvPoint p2;
				p2.x = skeletonData._2dPoint[j+2].x;
				p2.y = skeletonData._2dPoint[j+2].y;
				CvPoint p3;
				p3.x = skeletonData._2dPoint[j+6].x;
				p3.y = skeletonData._2dPoint[j+6].y;
				cvLine(image,p1,p2,cvScalar(0,0,225),2,8,0);
				cvLine(image,p1,p3,cvScalar(0,0,225),2,8,0);
			}
		}
		
		if (j==7 || j==11)
		{
			int length = 40;
			CvPoint start, end;
			start.x = p1.x-length>0?p1.x-length:0;
			start.y = p1.y-length>0?p1.y-length:0;

			end.x = p1.x+length<image->width-1?p1.x+length:image->width-1;
			end.y = p1.y+length<image->height-1?p1.y+length:image->height-1;
			if (showImage)
			{
				cvRectangle(image,start,end,cvScalar(255,0,0),2,8,0);
			}
			
			if (j==7)
			{
				CvRect leftRect;
				leftRect.x = start.x;
				leftRect.y = start.y;

				leftRect.width = end.x - start.x;
				leftRect.height = end.y - start.y;
				//cvSetImageROI(image,leftRect);
				cvSetImageROI(image,leftHand);
				cvResize(image, leftHandImg, 1);
				cvResetImageROI(image);
				IplImage* leftHandImg_ = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,1);
				cvCvtColor(leftHandImg,leftHandImg_,CV_BGR2GRAY);
				cvSaveImage(leftHandFileName,leftHandImg_);
				GetHOGHistogram_Patch(leftHandImg_,hog_hist_left);
				cvReleaseImage(&leftHandImg_);
			}
			if (j==11)
			{
				CvRect rightRect;
				rightRect.x = start.x;
				rightRect.y = start.y;

				rightRect.width = end.x - start.x;
				rightRect.height = end.y - start.y;
				//cvSetImageROI(image,rightRect);
				cvSetImageROI(image,rightHand);
				cvResize(image, rightHandImg, 1);
				cvResetImageROI(image);
				IplImage* rightHandImg_ = cvCreateImage(cvSize(PostureSIZE, PostureSIZE),8,1);
				cvCvtColor(rightHandImg,rightHandImg_,CV_RGB2GRAY);
				cvSaveImage(rightHandFileName,rightHandImg_);
				GetHOGHistogram_Patch(rightHandImg_,hog_hist_right);
				cvReleaseImage(&rightHandImg_);
			}


		}

	}
	cvReleaseImage(&leftHandImg);
	cvReleaseImage(&rightHandImg);

		//连接左右手的HOG.
	if (!hog_hist_right.empty() && !hog_hist_left.empty())
	{
		for (int i=0; i<hog_hist_left.size(); i++)
		{
			hog_hist.push_back(hog_hist_left[i]);
		}
		for (int i=0; i<hog_hist_right.size(); i++)
		{
			hog_hist.push_back(hog_hist_right[i]);
		}

		return true;
	}
	else
	{
		return false;
	}
}


// void LBP(IplImage* src, IplImage* dst)
// {
// 	int width=src->width;
// 	int height=src->height;
// 	for(int j=1;j<width-1;j++)
// 	{
// 		for(int i=1;i<height-1;i++)
// 		{
// 			uchar neighborhood[8]={0};
// 			neighborhood[7]	= CV_IMAGE_ELEM( src, uchar, i-1, j-1);
// 			neighborhood[6]	= CV_IMAGE_ELEM( src, uchar, i-1, j);
// 			neighborhood[5]	= CV_IMAGE_ELEM( src, uchar, i-1, j+1);
// 			neighborhood[4]	= CV_IMAGE_ELEM( src, uchar, i, j-1);
// 			neighborhood[3]	= CV_IMAGE_ELEM( src, uchar, i, j+1);
// 			neighborhood[2]	= CV_IMAGE_ELEM( src, uchar, i+1, j-1);
// 			neighborhood[1]	= CV_IMAGE_ELEM( src, uchar, i+1, j);
// 			neighborhood[0]	= CV_IMAGE_ELEM( src, uchar, i+1, j+1);
// 			uchar center = CV_IMAGE_ELEM( src, uchar, i, j);
// 			uchar temp=0;
// 
// 			for(int k=0;k<8;k++)
// 			{
// 				// temp+=(neighborhood[k]>center)*(1<<k);
// 				temp+=(neighborhood[k]>center)<<k;
// 			}
// 			CV_IMAGE_ELEM( dst, uchar, i, j)=temp;
// 		}
// 	}
// }

//基于旧版本的opencv的LBP算法opencv1.0

void LBP (IplImage *src,IplImage *dst)
{
	int tmp[8]={0};
	CvScalar s;

	IplImage * temp = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U,1);
	uchar *data=(uchar*)src->imageData;
	int step=src->widthStep;

	//cout<<"step"<<step<<endl;

	for (int i=1;i<src->height-1;i++)
		for(int j=1;j<src->width-1;j++)
		{
			int sum=0;
			if(data[(i-1)*step+j-1]>data[i*step+j])
				tmp[0]=1;
			else
				tmp[0]=0;
			if(data[i*step+(j-1)]>data[i*step+j])
				tmp[1]=1;
			else
				tmp[1]=0;
			if(data[(i+1)*step+(j-1)]>data[i*step+j])
				tmp[2]=1;
			else
				tmp[2]=0;
			if (data[(i+1)*step+j]>data[i*step+j])
				tmp[3]=1;
			else
				tmp[3]=0;
			if (data[(i+1)*step+(j+1)]>data[i*step+j])
				tmp[4]=1;
			else
				tmp[4]=0;
			if(data[i*step+(j+1)]>data[i*step+j])
				tmp[5]=1;
			else
				tmp[5]=0;
			if(data[(i-1)*step+(j+1)]>data[i*step+j])
				tmp[6]=1;
			else
				tmp[6]=0;
			if(data[(i-1)*step+j]>data[i*step+j])
				tmp[7]=1;
			else
				tmp[7]=0;	
			//计算LBP编码
			s.val[0]=(tmp[0]*1+tmp[1]*2+tmp[2]*4+tmp[3]*8+tmp[4]*16+tmp[5]*32+tmp[6]*64+tmp[7]*128);

			cvSet2D(dst,i,j,s);//写入LBP图像
		}
}


void mergeKeyFrame(vector<int> &flags, int mergeSpace, int deleteSegNum)
{
	//int mergeSpace = 1;   //可以被填补的空格数
	//int deleteSegNum = 0;  //小于等于这个帧数，该seg将要被删除
	int flagSize = flags.size();
	vector<int> tempFlags;

	for (int i=0; i<flagSize; i++)
	{
		tempFlags.push_back(flags[i]);
	}


	for (int i=mergeSpace; i<flagSize-mergeSpace; i++)
	{
		int leftID = 0;
		int rightID = flagSize - 1;
		for (int left = i-1; left>=0; left--)
		{
			if (tempFlags[left] == 1)
			{
				leftID = left;
				break;
			}
		}
		for (int right = i+1; right<flagSize; right++)
		{
			if (tempFlags[right]==1)
			{
				rightID = right;
				break;
			}
		}
		if ((rightID - leftID)-1 <= mergeSpace)
		{
			flags[i] = 1;
		}
	}


	int sp1 = 0;
	int sp2 = 0;
	while (sp1<flagSize)
	{
		if (flags[sp1] == 1)
		{
			for (int j=sp1+1; j<flagSize; j++)
			{
				if (flags[j] == 0)
				{
					sp2 = j;
					break;
				}
			}
			if ((sp2>=sp1) && (sp2 - sp1)<=deleteSegNum)
			{
				for (int k=sp1; k<sp2; k++)
				{
					flags[k] = 0;
				}
				sp1 = sp2;
			}
			if (sp2<sp1)
			{
				for (int k=sp1; k<flagSize; k++)
				{
					flags[k] = 0;
				}
			}
		}
		sp1++;
	}
}

void GetLBPfeature(IplImage* image, vector<float> &lbpFeature)
{
	

	IplImage* gray_plane = cvCreateImage(cvGetSize(image),8,1);
	cvCvtColor(image,gray_plane,CV_BGR2GRAY);
	IplImage* gray_plane_size = cvCreateImage(cvSize(50,50),8,1);
	cvResize(gray_plane,gray_plane_size);
	cvReleaseImage(&gray_plane);

	int width = gray_plane_size->width;
	int height = gray_plane_size->height;

	int nWidth = 10;
	int nHeight = 10;

	int bWidth = width/nWidth;
	int bHeight = height/nHeight;

	for (int i=0; i<bHeight; i++)
	{
		for (int j=0; j<bWidth; j++)
		{
			CvRect myRoi;
			myRoi.x = i*nHeight;
			myRoi.y = j*nWidth;
			myRoi.width = nWidth;
			myRoi.height = nHeight;
			cvSetImageROI(gray_plane_size,myRoi);

			int hist_size = 5;
			float range[] = {0,255};  //灰度级的范围
			float* ranges[]={range};
			CvHistogram* gray_hist = cvCreateHist(1,&hist_size,CV_HIST_ARRAY,ranges,1);
			cvCalcHist(&gray_plane_size,gray_hist,0,0);
			cvNormalizeHist(gray_hist,1.0);

			for(int i=0;i<hist_size;i++)
			{
				float bin_val = ((float)cvGetReal1D( (gray_hist)->bins, (i)));  
				//cout<<i<<" "<<bin_val<<endl;
				lbpFeature.push_back(bin_val);
			}
			cvResetImageROI(gray_plane_size);
		}
	}

}