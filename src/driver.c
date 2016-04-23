#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

void resize_driver(char *listfile, char *folder, int w, int h)
{
    int i,j,k;
    list *plist = get_paths(listfile);
    char **paths = (char **)list_to_array(plist);
    for (i=0;i<plist->size;i++)
    {
        if ((i%1000)==0)
            printf("progress = %d of %d\n", i, plist->size);
        char *path = paths[i];
        image img = load_image_color(path, w, h);
        char fname[512];
        int n = strlen(path);
        for (j=n-1;j>=0;j--)
            if (path[j]=='/')
            {
                k = j+1;
                for (j++;j<n-4;j++)
                    fname[j-k] = path[j];
                fname[j-k] = 0;
                break;
            }
        char outname[512];
        sprintf(outname, "%s/%s", folder, fname);
        save_image(img, outname);
    }

}

void train_driver(char *cfgfile, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "backup";
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }

    {
        char buff[256];
        sprintf(buff, "%s/%s.txt", backup_directory, base);
        freopen(buff, "w", stdout);
    }

    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    data train;

    fprintf(stderr, "Loading training labels\n");

    int i,j,k;
    FILE *fp = fopen("data/driver_imgs_list.csv", "r");
    if(!fp) file_error("data/driver_imgs_list.csv");
    char tmp[512];
    int *lbl = malloc(65536*2*4);
    memset(lbl, 0xff, 65536*2*4);
    fscanf(fp, "%s", tmp);
    int N = 0;
    while (fscanf(fp, "%s", tmp)==1)
    {
        int cl = tmp[6]-'0';
        int num = tmp[12]-'0';
        i = 13;
        while (tmp[i]!='.')
        {
            num = num*10 + (tmp[i]-'0');
            i++;
        }
        ++N;
        lbl[num] = cl;
//        fprintf(stderr, "%d: %d %d\n", count, cl, num);
    }
    fclose(fp);

    train.shallow = 0;
    matrix X = make_matrix(N, net.w*net.h*3);
    matrix y = make_matrix(N, 10);
    train.X = X;
    train.y = y;

    fprintf(stderr, "Loading training images\n");
    list *plist = get_paths("data/driver_imgs128.txt");
    char **paths = (char **)list_to_array(plist);
    for (i=0;i<plist->size;i++)
    {
      //  if (i>=5000) break;
        if ((i%1000)==0)
            fprintf(stderr, "progress = %d of %d\n", i, plist->size);
        char *path = paths[i];
        image img = load_image_color(path, net.w, net.h);
        char fname[512];
        int n = strlen(path);
        for (j=n-1;j>=0;j--)
            if (path[j]=='_')
            {
                k = j+1;
                int num = path[k]-'0';
                k++;
                while (path[k]!='.')
                {
                    num = num*10 + (path[k]-'0');
                    k++;
                }
                if (num<0 || num>100000)
                {
                  //  fprintf(stderr, "F =  %d %d\n", num, i);
                }
                if (lbl[num]<0 || lbl[num]>9)
                    fprintf(stderr, "ERROR = %d  %d %d\n", lbl[num], num, i);

                y.vals[i][(int)lbl[num]] = 1;
                for(j = 0; j < X.cols; ++j){
                    X.vals[i][j] = (double)img.data[j];
                }
                break;
            }
        free_image(img);
    }
    scale_data_rows(train, 1./255);

    data* theData = split_data(train, N*0.9, N);

    clock_t time=clock();
    float a[4];

    char backup_net[256];
    int nanCount = 0;

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

        float loss = train_network_sgd(net, theData[0], 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        if(get_current_batch(net)%100 == 0)
        {
            fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(N*0.9), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(N*0.9), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fflush(stdout);
            time=clock();
        }
        if (isnan(loss) || isnan(avg_loss))
        {
            // NaN detected!!!
            free_network(net);
            load_weights(&net, backup_net);
            nanCount++;
            if (nanCount>=5) break;
            continue;
        }
        if(get_current_batch(net)%1000 == 0){
            float *acc = network_accuracies(net, theData[1], 2);
            a[2] = acc[0];
            a[3] = acc[1];
            float mse1 = acc[2];
            float *accT = network_accuracies(net, theData[0], 2);
            a[0] = accT[0];
            a[1] = accT[1];
            float mse2 = accT[2];
            fprintf(stderr, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse2, a[2], a[3], mse1);
            fprintf(stdout, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse2, a[2], a[3], mse1);
            fflush(stdout);
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, get_current_batch(net));
            sprintf(backup_net, "%s/%s_%d.weights",backup_directory,base, get_current_batch(net));
            save_weights(net, buff);
            nanCount = 0;
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_data(train);
}

void testpoints_driver(char *cfgfile, char *weightfile, char *folder, char *imglist, char *outfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));
    freopen(outfile, "w", stdout);

    float *dat = malloc(net.w*net.h*3*16*sizeof(float));

    char tmp[512], tmp2[512];
    char* fname[16];
    int i,j,k;
    for (i=0;i<16;i++) fname[i] = malloc(512);
    int nidx = 0;
    list *plist = get_paths(imglist);
    char **paths = (char **)list_to_array(plist);
    for (i=0;i<plist->size;i++)
    {
        sscanf(paths[i],"%[^,],%s\n",tmp2,tmp);
        sprintf(tmp, "%s/%s", folder, tmp2);
        image img = load_image_color(tmp, net.w, net.h);
        for(j = 0; j < net.w*net.h*3; ++j){
            dat[j+nidx*net.w*net.h*3] = ((double)img.data[j]) / 255;
        }
        nidx++;
        if (nidx==1)
        {
            float *p = network_predict_gpu(net, dat);
            int kidx = 0;
            for (k=0;k<nidx;k++)
            {
                fprintf(stdout, "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", tmp2,
                        p[0+kidx],p[1+kidx],p[2+kidx],p[3+kidx],p[4+kidx],p[5+kidx],p[6+kidx],p[7+kidx]);
                kidx+=10;
            }
            nidx = 0;
        }
        free_image(img);
    }
}

void trainpoints_driver(char *cfgfile, char *folder, char *imglist, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "backup";
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }

    {
        char buff[256];
        sprintf(buff, "%s/%s.txt", backup_directory, base);
        freopen(buff, "w", stdout);
    }

    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);



    data train;

    fprintf(stderr, "Loading training labels\n");

    char tmp[512];
    char tmp2[512];
    int i,j,k;
    FILE *fp = fopen(imglist, "r");
    if(!fp) file_error(imglist);
    // read header
  //  fscanf(fp, "%s\n", tmp);
    train.shallow = 0;
    matrix X = make_matrix(290, net.w*net.h*3);
    matrix y = make_matrix(290, 2*4);
    train.X = X;
    train.y = y;

    int YSZ = 0;
    float fnum[16];
    while (1)
    {
        if (feof(fp)) break;
        fscanf(fp, "%[^,],%f,%f,%f,%f,%f,%f,%f,%f%s\n", tmp, &fnum[0], &fnum[1],&fnum[2],&fnum[3],&fnum[4],&fnum[5],&fnum[6],&fnum[7],tmp2);
        if (fnum[0]>=0)
        {
           // fprintf(stderr, "%s\n", tmp);
            for (i=0;i<8;i++)
                y.vals[YSZ][i] = fnum[i];

            sprintf(tmp2, "%s%s", folder, tmp);
            image img = load_image_color(tmp2, net.w, net.h);
            for(j = 0; j < X.cols; ++j){
                X.vals[YSZ][j] = (double)img.data[j];
            }
            free_image(img);
            YSZ++;
        }
    }
  //  X = resize_matrix(X, YSZ);
  //  y = resize_matrix(y, YSZ);
    train.X = X;
    train.y = y;
    scale_data_rows(train, 1./255);
    fprintf(stderr, "Training size = %d\n", YSZ);
    int N = YSZ;

    data* theData = split_data(train, YSZ*0.9, YSZ);
    fclose(fp);


    clock_t time=clock();
    float a[4];

    char backup_net[256];
    int nanCount = 0;

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

        float loss = train_network_sgd(net, theData[0], 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        if(get_current_batch(net)%10 == 0)
        {
            fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(N*0.9), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(N*0.9), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fflush(stdout);
            time=clock();
        }
        if (isnan(loss) || isnan(avg_loss))
        {
            // NaN detected!!!
            free_network(net);
            load_weights(&net, backup_net);
            nanCount++;
            if (nanCount>=5) break;
            continue;
        }
        if(get_current_batch(net)%100 == 0){
            matrix testdt = network_predict_data(net, theData[1]);
            float mse1 = matrix_mse(theData[1].y, testdt);
            matrix traindt = network_predict_data(net, theData[0]);
            float mse2 = matrix_mse(theData[0].y, traindt);

            fprintf(stderr, "MSE: train(%f) test(%f)\n", mse2, mse1);
            fprintf(stdout, "MSE: train(%f) test(%f)\n", mse2, mse1);
            fflush(stdout);
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, get_current_batch(net));
            sprintf(backup_net, "%s/%s_%d.weights",backup_directory,base, get_current_batch(net));
            save_weights(net, buff);
            nanCount = 0;
            free_matrix(testdt);
            free_matrix(traindt);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_data(train);
}

void test_driver(char *filename, char *weightfile, char *listfile, int w, int h)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));


    clock_t time;
    freopen("driver_predict.txt", "w", stdout);
    fprintf(stdout, "img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n");

    float *dat = malloc(net.w*net.h*3*16*sizeof(float));
    //matrix X = make_matrix(16, 64*64*3);

    char* fname[16];
    // TODO: change for larger batches
    // TODO: augmentations
    int i,j,k;
    for (i=0;i<16;i++) fname[i] = malloc(512);
    int nidx = 0;
    list *plist = get_paths(listfile);
    char **paths = (char **)list_to_array(plist);
    for (i=0;i<plist->size;i++)
    {
        if ((i%1000)==0)
         fprintf(stderr, "progress = %d of %d\n", i, plist->size);

        char *path = paths[i];
        int n = strlen(path);
        for (j=n-1;j>=0;j--)
            if (path[j]=='/')
            {
                k = j+1;
                for (j++;j<n;j++)
                    fname[nidx][j-k] = path[j];
                fname[nidx][j-k] = 0;
                break;
            }
      //  fprintf(stderr, "Loading image %s %d %d\n", path,w, h);
        image img = load_image_color(path, w, h);
        for(j = 0; j < net.w*net.h*3; ++j){
            dat[j+nidx*net.w*net.h*3] = ((double)img.data[j]) / 255;
        }
       // printf("predicting\n");
        nidx++;
        if (nidx==1)
        {
            float *p = network_predict_gpu(net, dat);
            int kidx = 0;
            for (k=0;k<nidx;k++)
            {
//                int mxj = 0;
//                for (j=1;j<10;j++)
//                {
//                    if (p[j+kidx]>p[mxj+kidx]) mxj = j;
//                }
//                if (p[mxj+kidx]>0.9)
//                {
//                    for (j=0;j<10;j++)
//                    {
//                        if (j==mxj)
//                            p[j+kidx] = 1.0;
//                        else
//                            p[j+kidx] = 0.0;
//                    }
//                }
                //printf("done\n");
                fprintf(stdout, "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", fname[k],
                        p[0+kidx],p[1+kidx],p[2+kidx],p[3+kidx],p[4+kidx],p[5+kidx],p[6+kidx],p[7+kidx],p[8+kidx],p[9+kidx] );
                kidx+=10;
            }
            nidx = 0;
        }
        free_image(img);
    }
    if (nidx>0)
    {
        float *p = network_predict_gpu(net, dat);
        int kidx = 0;
        for (k=0;k<nidx;k++)
        {
//            int mxj = 0;
//            for (j=1;j<10;j++)
//            {
//                if (p[j+kidx]>p[mxj+kidx]) mxj = j;
//            }
//            for (j=0;j<10;j++)
//            {
//                if (j==mxj)
//                    p[j+kidx] = 1.0;
//                else
//                    p[j+kidx] = 0.0;
//            }
            //printf("done\n");
            fprintf(stdout, "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", fname[k],
                    p[0+kidx],p[1+kidx],p[2+kidx],p[3+kidx],p[4+kidx],p[5+kidx],p[6+kidx],p[7+kidx],p[8+kidx],p[9+kidx] );
            kidx+=10;
        }
    }

    time=clock();

//    float *acc = network_accuracies(net, test, 2);
//    avg_acc += acc[0];
//    avg_top5 += acc[1];
//    printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock()-time), test.X.rows);

//    save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);

//    free_data(test);
}

void test_driver_csv(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    data test;
    //test = load_driver_data("data/driver/test_X.bin", "data/driver/test_y.bin", 800*10);

    matrix pred = network_predict_data(net, test);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        flip_image(im);
    }
    matrix pred2 = network_predict_data(net, test);
    scale_matrix(pred, .5);
    scale_matrix(pred2, .5);
    matrix_add_matrix(pred2, pred);

    matrix_to_csv(pred);
    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
}

void test_driver_csvtrain(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    data test;
    //test = load_driver_data("data/driver/test_X.bin", "data/driver/test_y.bin", 800*10);

    matrix pred = network_predict_data(net, test);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(32, 32, 3, test.X.vals[i]);
        flip_image(im);
    }
    matrix pred2 = network_predict_data(net, test);
    scale_matrix(pred, .5);
    scale_matrix(pred2, .5);
    matrix_add_matrix(pred2, pred);

    matrix_to_csv(pred);
    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
}

void run_driver(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [resize] [imagelist] [destination folder] width height \n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [test] [cfg] [weights] [imagelist] width height\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [trainpoints] [cfg] [imagefolder] [imagelist] [weights]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [testpoints] [cfg] [weights] [imagefolder] [imagelist] [outfile]\n", argv[0], argv[1]);
        return;
    }

    int w,h;
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_driver(cfg, weights);
    else if(0==strcmp(argv[2], "test"))
    {
        char *listfile = argv[5];
        w = atoi(argv[6]);
        h = atoi(argv[7]);
        test_driver(cfg, weights, listfile, w, h);
    }
    else if(0==strcmp(argv[2], "resize"))
    {
        w = atoi(argv[5]);
        h = atoi(argv[6]);
        resize_driver(cfg, weights, w, h);
    }
    else if(0==strcmp(argv[2], "trainpoints"))
    {
        trainpoints_driver(cfg, weights, argv[5], (argc>6 ? argv[6] : 0));
    }
    else if(0==strcmp(argv[2], "testpoints"))
    {
        testpoints_driver(cfg, weights, argv[5], argv[6], argv[7]);
    }

}


