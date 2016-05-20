#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;


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

string header;

map< string,vector<double> > loadPredictions(char *fname)
{
    cout << "loading " << fname << endl;
    map<string, vector<double> > dt;
    char buf[65536];
    ifstream fi(fname);
    fi.getline(buf, 65536);
    header = buf;
    while (!fi.eof())
    {
        fi.getline(buf, 65536);
        string s = buf;
        if (s.length()==0) break;
        vector<string> tok = tokenize(s, ',');
        if (tok.size()==0) break;
        vector<double> v;
        for (int i=1;i<tok.size();i++)
            v.push_back(atof(tok[i].c_str()));
        dt[tok[0]] = v;
    }
    fi.close();
    return dt;
}

void savePredictions(char *fname, map< string, vector<double> > &dt)
{
    ofstream fo(fname);
    fo << header << endl;
    for (map< string, vector<double> >::iterator itr=dt.begin();itr!=dt.end();++itr)
    {
        fo << itr->first;
        for (int i=0;i<itr->second.size();i++)
            fo << "," << itr->second[i];
        fo << endl;
    }
    fo.close();
}

int main( int argc, char** argv )
{
    if (argc<3)
    {
        cout << "Usage: ensemble file1 file2 ... \n";
        return 0;
    }

    map< string, vector<double> > final;

    final = loadPredictions(argv[1]);
    int cnt = 1;
    for (int i=2;i<argc;i++)
    {
        cnt++;
        map< string, vector<double> > dt = loadPredictions(argv[i]);
        for (map< string, vector<double> >::iterator itr=dt.begin();itr!=dt.end();++itr)
        {
            vector<double> &src = itr->second;
            vector<double> &dest = final[itr->first];
            for (int i=0;i<dest.size() && i<src.size();i++)
            {
                dest[i] += src[i];
            }
        }
    }
    // average
    for (map< string, vector<double> >::iterator itr=final.begin();itr!=final.end();++itr)
    {
        vector<double> &dest = final[itr->first];
        for (int i=0;i<dest.size();i++)
        {
            dest[i] /= cnt;
            if (dest[i]<0.01) dest[i] = 0;
            if (dest[i]>0.99) dest[i] = 1.0;
        }
    }

    savePredictions("ensemble.txt", final);


    return 0;
}

