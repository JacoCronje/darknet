#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
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

void mergeFold(int argc, char *argv[])
{
    char buf[65536];

    string outFile = argv[2];
    vector<int> iter;
    vector<double> dt[2048];
    vector<int> totdt[2048];
    int n = argc-3;
    cout << "Creating " << outFile << endl;
    int lasti = 0;
    for (int i=0;i<n;i++)
    {
        for (int fold=0;fold<10;fold++)
        {
            stringstream ss;
            ss << argv[i+3] << fold << ".txt";
            cout << "Opening " << ss.str() << endl;
            ifstream fi(ss.str().c_str());
            int widx = 0;
            while (!fi.eof())
            {
                fi.getline(buf, 65536);
                string s = buf;
                int i1 = s.find("Accuracy");
                if (i1!=string::npos)
                {
                    //cout << s << endl;
                    int i2 = s.find("train(");
                    int i3 = s.find(") test(");
                    string s1 = s.substr(i2+6, i3-i2-6);
                    string s2 = s.substr(i3+7, s.length()-i3-8);
                    vector<string> tok1 = tokenize(s1, ' ');
                    vector<string> tok2 = tokenize(s2, ' ');
                    if (fold==0)
                    {
                        if (i==0)
                        {
                            iter.push_back(lasti);
                        }
                        dt[i*6].push_back(atof(tok1[0].c_str()));
                        dt[i*6+1].push_back(atof(tok1[1].c_str()));
                        dt[i*6+2].push_back(atof(tok1[2].c_str()));
                        dt[i*6+3].push_back(atof(tok2[0].c_str()));
                        dt[i*6+4].push_back(atof(tok2[1].c_str()));
                        dt[i*6+5].push_back(atof(tok2[2].c_str()));
                        for (int k=0;k<6;k++)
                            totdt[i*6+k].push_back(1);
                    } else
                    {
                        dt[i*6][widx]+=(atof(tok1[0].c_str()));
                        dt[i*6+1][widx]+=(atof(tok1[1].c_str()));
                        dt[i*6+2][widx]+=(atof(tok1[2].c_str()));
                        dt[i*6+3][widx]+=(atof(tok2[0].c_str()));
                        dt[i*6+4][widx]+=(atof(tok2[1].c_str()));
                        dt[i*6+5][widx]+=(atof(tok2[2].c_str()));
                        for (int k=0;k<6;k++)
                            totdt[i*6+k][widx]++;
                        widx++;
                    }
                } else
                {
                    vector<string> tok = tokenize(s, ',');
                    lasti = atoi(tok[0].c_str());
    //                if (i==0) iter.push_back(lasti);
                }
            }
            fi.close();
        }
    }
    ofstream fo(outFile.c_str());
    fo << "Iter, ";
    for (int i=0;i<n;i++)
    {
        string s = argv[i+3];
        int i1 = s.find_last_of("/");
        if (i1!=string::npos)
        {
            s = s.substr(i1+1, s.length()-i1-1);
        }
        fo << "train_top1_";
        fo << s << ", ";
        fo << "train_top5_";
        fo << s << ", ";
        fo << "train_mse_";
        fo << s << ", ";
        fo << "test_top1_";
        fo << s << ", ";
        fo << "test_top5_";
        fo << s << ", ";
        fo << "test_mse_";
        fo << s;
        if (i<n-1)
            fo << ", ";
    }
    fo << endl;
    for (int i=0;i<dt[0].size();i++)
    {
        fo << iter[i];
        for (int j=0;j<n*6;j++)
            fo << ", " << dt[j][i]/totdt[j][i];
        fo << endl;
    }

    fo.close();

}

int main(int argc, char *argv[])
{
    char buf[65536];

    if (argc<3)
    {
        cout << "Usage: ./preparePlot output.csv input1.csv input2.csv .... inputN.csv" << endl;
        return 0;
    }
    string outFile = argv[1];
    if (outFile=="fold")
    {
        mergeFold(argc, argv);
        return 0;
    }

    vector<int> iter;
    vector<double> dt[2048];
    int n = argc-2;
    cout << "Creating " << outFile << endl;
    int lasti = 0;
    for (int i=0;i<n;i++)
    {
        cout << "Opening " << argv[i+2] << endl;
        ifstream fi(argv[i+2]);
        while (!fi.eof())
        {
            fi.getline(buf, 65536);
            string s = buf;
            int i1 = s.find("Accuracy");
            if (i1!=string::npos)
            {
                //cout << s << endl;
                int i2 = s.find("train(");
                int i3 = s.find(") test(");
                string s1 = s.substr(i2+6, i3-i2-6);
                string s2 = s.substr(i3+7, s.length()-i3-8);
                vector<string> tok1 = tokenize(s1, ' ');
                vector<string> tok2 = tokenize(s2, ' ');
                if (i==0)
                {
                    iter.push_back(lasti);
                }
                dt[i*6].push_back(atof(tok1[0].c_str()));
                dt[i*6+1].push_back(atof(tok1[1].c_str()));
                dt[i*6+2].push_back(atof(tok1[2].c_str()));
                dt[i*6+3].push_back(atof(tok2[0].c_str()));
                dt[i*6+4].push_back(atof(tok2[1].c_str()));
                dt[i*6+5].push_back(atof(tok2[2].c_str()));
            } else
            {
                vector<string> tok = tokenize(s, ',');
                lasti = atoi(tok[0].c_str());
//                if (i==0) iter.push_back(lasti);
            }
        }
        fi.close();
    }
    ofstream fo(outFile.c_str());
    fo << "Iter, ";
    for (int i=0;i<n;i++)
    {
        string s = argv[i+2];
        int i1 = s.find_last_of("/");
        if (i1!=string::npos)
        {
            s = s.substr(i1+1, s.length()-i1-1-4);
        }
        fo << "train_top1_";
        fo << s << ", ";
        fo << "train_top5_";
        fo << s << ", ";
        fo << "train_mse_";
        fo << s << ", ";
        fo << "test_top1_";
        fo << s << ", ";
        fo << "test_top5_";
        fo << s << ", ";
        fo << "test_mse_";
        fo << s;
        if (i<n-1)
            fo << ", ";
    }
    fo << endl;
    for (int i=0;i<dt[0].size();i++)
    {
        fo << iter[i];
        for (int j=0;j<n*6;j++)
            fo << ", " << dt[j][i];
        fo << endl;
    }

    fo.close();






    return 0;
}

