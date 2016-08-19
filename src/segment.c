#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

double network_score(network net, data test)
{
    double score = 0;
    int i,j,b;
    int k = get_network_output_size(net);
   // fprintf(stderr, "k=%d rows=%d cols=%d batch=%d\n", k, test.X.rows, test.X.cols, net.batch);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b)
        {
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j)
            {
                double err = out[j+b*k] - test.y.vals[i+b][j];
                score += err*err;
            }
        }
    }
    free(X);
    score /= (test.X.rows*k);
    return score;
}


void train_segment(char *cfgfile, char* listfile, char *weightfile,  int fold)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *baseO = basecfg(cfgfile);
    char base[512];
    if (fold<0)
        sprintf(base, "%s", baseO);
    else
        sprintf(base, "%s%d", baseO, fold);

    char *backup_directory = "backup";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile)
    {
        load_weights(&net, weightfile);
    }

    char buff[256];
    sprintf(buff, "%s/%s.txt", backup_directory, base);
    freopen(buff, "w", stdout);

    int Ntrain = 1000;
    int Ntest = 100;

    data train, test;
    train.shallow = 0;
    train.X = make_matrix(Ntrain, net.w*net.h*net.c);
    train.y = make_matrix(Ntrain, net.w*net.h*net.c);
    test.shallow = 0;
    test.X = make_matrix(Ntest, net.w*net.h*net.c);
    test.y = make_matrix(Ntest, net.w*net.h*net.c);
    int itrain = 0;
    int itest = 0;

    srand(12357+fold);

    int i,j,k;
    list *plist = get_paths(listfile);
    char **paths = (char **)list_to_array(plist);
    for (i=0;i<plist->size;i++)
    {

       // if ((i%1000)==0)
         //   printf("progress = %d of %d\n", i, plist->size);
        char *path = paths[i];
        int n = strlen(path);
        if (path[n-5]=='k') continue;

        char ifname[512];
        sprintf(ifname, "/home/jcronje/dev/kaggle/sonar/train/%s", path);

        if (itrain>=Ntrain && itest>=Ntest) break;

        if (rand()%(Ntrain+Ntest)<Ntrain)
        {
            // training
            if (itrain>=Ntrain) continue;
            if ((itrain%100)==0)
            fprintf(stderr, "Loading training %d %s\n", itrain, ifname);
            image img = load_image(ifname, net.w, net.h, net.c);
            for(j = 0; j < train.X.cols; ++j){
                train.X.vals[itrain][j] = (float)(img.data[j]);
            }
            free_image(img);
            n = strlen(ifname);
            ifname[n-4] = '_';
            ifname[n-3] = 'm';
            ifname[n-2] = 'a';
            ifname[n-1] = 's';
            ifname[n] = 'k';
            ifname[n+1] = '.';
            ifname[n+2] = 't';
            ifname[n+3] = 'i';
            ifname[n+4] = 'f';
            ifname[n+5] = 0;
            img = load_image(ifname, net.w, net.h, net.c);
            float pixsum = 0;
            for(j = 0; j < train.y.cols; ++j){
                train.y.vals[itrain][j] = (float)(img.data[j]);
                pixsum += train.y.vals[itrain][j];
            }
            free_image(img);
            if (pixsum<0.5) continue;
            itrain++;
        } else
        {
            // testing
            if (itest>=Ntest) continue;
            if ((itest%10)==0)
            fprintf(stderr, "Loading testing %d %s\n", itest, ifname);
            image img = load_image(ifname, net.w, net.h, net.c);
            for(j = 0; j < test.X.cols; ++j){
                test.X.vals[itest][j] = (float)(img.data[j]);
            }
            free_image(img);
            n = strlen(ifname);
            ifname[n-4] = '_';
            ifname[n-3] = 'm';
            ifname[n-2] = 'a';
            ifname[n-1] = 's';
            ifname[n] = 'k';
            ifname[n+1] = '.';
            ifname[n+2] = 't';
            ifname[n+3] = 'i';
            ifname[n+4] = 'f';
            ifname[n+5] = 0;
            img = load_image(ifname, net.w, net.h, net.c);
            float pixsum = 0;
            for(j = 0; j < test.y.cols; ++j){
                test.y.vals[itest][j] = (float)(img.data[j]);
                pixsum += test.y.vals[itest][j];
            }
            free_image(img);
            if (pixsum<0.5) continue;
            itest++;
        }
    }

    srand(time(0));

    clock_t time=clock();
    float a[4];

    char backup_net[256];
    int nanCount = 0;
    int N = Ntrain;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

        float loss = train_network_sgd(net, train, 1);


       //  save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);
        //  return 0;

        if(avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        if(get_current_batch(net)%10 == 0)
        {
            fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fflush(stdout);
            time=clock();
        }
        if (isnan(loss) || isnan(avg_loss))
        {
            // NaN detected!!!
            fprintf(stderr, "NAN DETECTED\n");
            fprintf(stdout, "NAN DETECTED\n");
            free_network(net);
            load_weights(&net, backup_net);
            nanCount++;
            if (nanCount>=5) break;
            continue;
        }
        if(get_current_batch(net)%100 == 0){
            double restest = network_score(net, test);
            double restrain = network_score(net, train);

            save_network_feature_maps(net, 0, net.n-2, "network", 10, 2);

            fprintf(stderr, "Accuracy: train %f test %f\n", restrain, restest);
            fprintf(stdout, "Accuracy: train %f test %f\n", restrain, restest);
            fflush(stdout);
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, get_current_batch(net));
            sprintf(backup_net, "%s/%s_%d.weights",backup_directory,base, get_current_batch(net));
            save_weights(net, buff);
            nanCount = 0;
        }
    }

    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free(baseO);
    free_data(train);
    free_data(test);

}

void test_segment(char *filename, char* listfile, char *weightfile)
{
//    network net = parse_network_cfg(filename);
//    if(weightfile){
//        load_weights(&net, weightfile);
//    }
//    srand(time(0));

//    clock_t time;
//    float avg_acc = 0;
//    float avg_top5 = 0;

//    data test;
//    test = load_segment_data("data/segment/test_X.bin", "data/segment/test_y.bin", 800*10);

//    time=clock();

//    float *acc = network_accuracies(net, test, 2);
//    avg_acc += acc[0];
//    avg_top5 += acc[1];
//    printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock()-time), test.X.rows);

//    save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);

//    free_data(test);
}


void run_segment(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [listfile] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")) train_segment(cfg,  argv[4], weights, -1);
    else if(0==strcmp(argv[2], "test")) test_segment(cfg, argv[4], weights);
}


