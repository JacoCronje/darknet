#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

IplImage* cropImage(const IplImage *img, const CvRect region);
IplImage* resizeImage(const IplImage *origImg, int newWidth,int newHeight, bool keepAspectRatio);

#define NP 10

struct SItem
{
    string imgname;
    vector<double> x,y;
};
vector<SItem> images;

vector<string> tokenize(const string& str, char delimiters)
{
    vector<string> tokens;
    int len = (int)str.length();

    int pos = 0;
    int i = 0;
    int quote = 0;
    for (;;)
    {
        while (pos<len && (str[pos]!=delimiters || (str[pos]==delimiters && quote!=0)))
        {
            if (str[pos]=='"')
                quote = 1-quote;
            pos++;
        }
        if (pos<len && str[pos]==delimiters)
        {
            if (str[i]=='"')
            {
                tokens.push_back(str.substr(i+1, pos-i-2));
            }
            else
            {
                tokens.push_back(str.substr(i, pos-i));
            }
            i = pos+1;
        }
        else
        {
            if (str[i]=='"')
            {
                tokens.push_back(str.substr(i+1, pos-i-2));
            }
            else
            {
                tokens.push_back(str.substr(i, pos-i));
            }
            break;
        }
        pos++;
    }
    return tokens;
}

void saveAnnotations(char *fname)
{
    ofstream fo(fname);
    for (int i=0;i<images.size();i++)
    {
        fo << images[i].imgname;
        for (int j=0;j<images[i].x.size();j++)
        {
            fo << "," << images[i].x[j] << "," << images[i].y[j];
        }
        fo << endl;
    }
    fo.close();
}

void loadAnnotations(char *fname)
{
    images.clear();
    char buf[65536];
    ifstream fi(fname);
    while (!fi.eof())
    {
        fi.getline(buf, 65536);
        string s = buf;
        if (s.length()==0) break;
        vector<string> tok = tokenize(s, ',');
        if (tok.size()==0) break;
        SItem it;
        it.imgname = tok[0];
        it.x.resize(NP, -1);
        it.y.resize(NP, -1);
        int j = 0;
        for (int i=1;i<tok.size();i+=2)
        {
            it.x[j] = atof(tok[i].c_str());
            it.y[j] = atof(tok[i+1].c_str());
            j++;
        }
        images.push_back(it);
    }
    fi.close();
}

void onmouse(int event, int x, int y, int flags, void* param)
{
    if (event==CV_EVENT_MOUSEMOVE)
    {
        int *pp = (int*)param;
        *pp = x+(y<<12);
    }
}


int main( int argc, char** argv )
{
    if (argc<4)
    {
        cout << "Usage: annotate [image folder] [image list] [output]\n";
        return 0;
    }

    loadAnnotations(argv[2]);

    IplImage *img = 0;
    IplImage *imgM = 0;
    bool pause = false;

    cvNamedWindow( "Annotate", 0);
    int mousepos = 0;
    setMouseCallback("Annotate", onmouse, &mousepos);

    int idx = 1;
    {
        stringstream ss;
        ss << argv[1] << images[idx].imgname;
        img = cvLoadImage(ss.str().c_str());
       // imgM = cvLoadImage(ss.str().c_str());
    }
    imgM = cvCreateImage(cvSize(512,512), IPL_DEPTH_8U, 3);


    vector<CvScalar> col;
    col.push_back(CV_RGB(255,0,0));
    col.push_back(CV_RGB(0,255,0));
    col.push_back(CV_RGB(0,0,255));
    col.push_back(CV_RGB(255,255,0));
    col.push_back(CV_RGB(0,255,255));
    col.push_back(CV_RGB(0,0,0));
    col.push_back(CV_RGB(0,0,0));
    col.push_back(CV_RGB(0,0,0));
    col.push_back(CV_RGB(0,0,0));
    col.push_back(CV_RGB(0,0,0));
    col.push_back(CV_RGB(0,0,0));

    while(1) {
        char c = cvWaitKey(33); // press escape to quit
        if (c==27) break;
        if (c==',')
        {
            cvReleaseImage(&img);
            idx = (idx-1+images.size())%images.size();
            stringstream ss;
            ss << argv[1] << images[idx].imgname;
            img = cvLoadImage(ss.str().c_str());
            saveAnnotations(argv[3]);
        }
        if (c=='.')
        {
            cvReleaseImage(&img);
            idx = (idx+1)%images.size();
            stringstream ss;
            ss << argv[1] << images[idx].imgname;
            img = cvLoadImage(ss.str().c_str());
            saveAnnotations(argv[3]);
        }
        if (c=='r')
        {
            for (int i=0;i<images.size();i++)
            {
                int i1 = rand()%images.size();
                swap(images[i], images[i1]);
            }
            cvReleaseImage(&img);
            stringstream ss;
            ss << argv[1] << images[idx].imgname;
            img = cvLoadImage(ss.str().c_str());
        }
        if (c>='1' && c<='9')
        {
            int mx = mousepos&4095;
            int my = mousepos>>12;
            int cidx = c-'1';
            double prevx = images[idx].x[cidx];
            double prevy = images[idx].y[cidx];
            images[idx].x[cidx] = (double)(mx) / imgM->width;
            images[idx].y[cidx] = (double)(my) / imgM->height;
            if (fabs(images[idx].x[cidx]-prevx)<0.00001 && fabs(images[idx].y[cidx]-prevy)<0.00001)
            {
                images[idx].x[cidx] = -1;
                images[idx].y[cidx] = -1;
            }
        }

        //cvCopy(img, imgM, NULL);
        cvResize(img, imgM);
        for (int i=0;i<NP;i++)
        {
            if (images[idx].x[i]<0) continue;
            int px = images[idx].x[i]*imgM->width;
            int py = images[idx].y[i]*imgM->height;
            cvRectangle( imgM, cvPoint(px-8,py-8), cvPoint(px+8,py+8), col[i], 3, 8, 0 );
        }


        cvShowImage( "Annotate", imgM );
        //cvResizeWindow( "Annotate", 512, 512);
        setMouseCallback("Annotate", onmouse, &mousepos);
    }

    cvReleaseImage(&img);
    return 0;
}

