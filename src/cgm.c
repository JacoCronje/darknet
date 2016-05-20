#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

int cgm_set = 10;


void train_cgm(char *cfgfile, char *trainlist, char *testlist)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "backup";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    {
        char buff[256];
        sprintf(buff, "%s/%s.txt", backup_directory, base);
        freopen(buff, "w", stdout);
    }

    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    int N = 0;

    int input_len = net.w;
    int stride = net.w/4;
    int i,j,k;
    data train;
    data test;
//TODO: load data
    {
        train.shallow = 0;
        train.X = make_matrix(15107, net.w*net.h*net.c);
        train.y = make_matrix(15107, 1);

        list *plist = get_paths(trainlist);
        char **paths = (char **)list_to_array(plist);
        float cbuf[4096*16];
        int cidx = 0;
        int widx = 0;
        int cnt = 0;
        int cntstride = 0;
        int freq[16] = {0};
        for (i=0;i<plist->size;i++)
        {
            FILE *fp = fopen(paths[i], "rb");
            if(!fp) file_error(paths[i]);
            while (!feof(fp))
            {
                unsigned short bytes[11];
                fread(bytes, 2, 11, fp);
                float fbytes[2];
                fread(fbytes, 4, 2, fp);
                // put into circular buffer
                for (j=0;j<10;j++)
                    cbuf[cidx+(j<<12)] = ((float)(bytes[j])-32768.f)/65536.f;
                cidx = (1+cidx)&4095;
                cnt++;
                cntstride++;
                //fbytes[0] = (float)(bytes[10]);// - 40.0;
                //fbytes[0] = 54.0+(fbytes[0]*18.0*10.0/80.0);
                // valid data to add ?
                if (fbytes[0]>54 && cnt>input_len && cntstride>stride)
                {
                    cntstride = 0;
//                    int clas = (fbytes[0]-54)/18;
//                    if (clas<0) clas = 0;
//                    if (clas>9) clas = 9;
                    train.y.vals[widx][0] = (fbytes[0]-50)/200.0;
                    //freq[clas]++;
                    //
                    for (k=0;k<net.w;k++)
                    for (j=0;j<10;j++)
                    {
                        train.X.vals[widx][(j*net.w)+k] = cbuf[(j<<12)+(((cidx-1-net.w+k)+4096)&4095)];
                    }
                    widx++;
                }
            }
            fclose(fp);
        }
        //translate_data_rows(d, -128);
        //normalize_data_rows(d);
        //scale_data_rows(train, 1./65536);
        fprintf(stderr, "widx = %d\n", widx);
        fprintf(stderr, "Freq = %d %d %d %d %d %d %d %d %d %d\n", freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6], freq[7], freq[8], freq[9]);
    }
    {
        test.shallow = 0;
        test.X = make_matrix(2277, net.w*net.h*net.c);
        test.y = make_matrix(2277, 1);

        list *plist = get_paths(testlist);
        char **paths = (char **)list_to_array(plist);
        float cbuf[4096*16];
        int cidx = 0;
        int widx = 0;
        int cnt = 0;
        int cntstride = 0;
        int freq[16] = {0};
        for (i=0;i<plist->size;i++)
        {
            FILE *fp = fopen(paths[i], "rb");
            if(!fp) file_error(paths[i]);
            while (!feof(fp))
            {
                unsigned short bytes[11];
                fread(bytes, 2, 11, fp);
                float fbytes[2];
                fread(fbytes, 4, 2, fp);
                // put into circular buffer
                for (j=0;j<10;j++)
                    cbuf[cidx+(j<<12)] = ((float)(bytes[j])-32768.f)/65536.f;
                cidx = (1+cidx)&4095;
                cnt++;
                cntstride++;
//                fbytes[0] = (float)(bytes[10]) - 40.0;
//                fbytes[0] = 54.0+(fbytes[0]*18.0*10.0/80.0);
//                // valid data to add ?
//                if (fbytes[0]>54 && cnt>input_len && cntstride>stride)
//                {
//                    cntstride = 0;
//                    int clas = (fbytes[0]-54)/18;
//                    if (clas<0) clas = 0;
//                    if (clas>9) clas = 9;
//                    test.y.vals[widx][clas] = 1;
//                    freq[clas]++;
              // fbytes[0] = (float)(bytes[10]);// - 40.0;
                //fbytes[0] = 54.0+(fbytes[0]*18.0*10.0/80.0);
                // valid data to add ?
                if (fbytes[0]>54 && cnt>input_len && cntstride>stride)
                {
                    cntstride = 0;
//                    int clas = (fbytes[0]-54)/18;
//                    if (clas<0) clas = 0;
//                    if (clas>9) clas = 9;
                    test.y.vals[widx][0] = (fbytes[0]-50)/200.0;
                    //
                    for (k=0;k<net.w;k++)
                    for (j=0;j<10;j++)
                    {
                        test.X.vals[widx][(j*net.w)+k] = cbuf[(j<<12)+(((cidx-1-net.w+k)+4096)&4095)];
                    }
                    widx++;
                }
            }
            fclose(fp);
        }
        //translate_data_rows(d, -128);
        //normalize_data_rows(d);
        //scale_data_rows(train, 1./65536);
        fprintf(stderr, "widx = %d\n", widx);
        fprintf(stderr, "Freq = %d %d %d %d %d %d %d %d %d %d\n", freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6], freq[7], freq[8], freq[9]);
    }

   // return;

    clock_t time=clock();
    float a[4];

    char backup_net[256];
    int nanCount = 0;

    N = train.X.rows;

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        if(get_current_batch(net)%10 == 0)
        {
            fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fflush(stdout);
            time=clock();
        }
//        if (isnan(loss) || isnan(avg_loss))
//        {
//            // NaN detected!!!
//            free_network(net);
//            load_weights(&net, backup_net);
//            nanCount++;
//            if (nanCount>=5) break;
//            continue;
//        }
        if(get_current_batch(net)%200 == 0){
            float *acc = network_accuracies(net, test, 2);
            a[2] = acc[0];
            a[3] = acc[1];
            float mse1 = acc[2];
            float *accT = network_accuracies(net, train, 2);
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
           // save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_data(train);
    free_data(test);
}


void test_cgm(char *cfgfile, char *weightfile, char *session)
{

    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    load_weights(&net, weightfile);

    freopen("out.txt", "w", stdout);

    int input_len = net.w;
    int stride = net.w/4;

    int i,j,k;

    data train;
    train.shallow = 0;
    train.X = make_matrix(128, net.w*net.h*net.c);
    train.y = make_matrix(128, 1);

    float cbuf[4096*16];
    int cidx = 0;
    int cnt = 0;
    int cntstride = 0;
    int freq[16] = {0};
    FILE *fp = fopen(session, "rb");
    if(!fp) file_error(session);
    while (!feof(fp))
    {
        unsigned short bytes[11];
        fread(bytes, 2, 11, fp);
        float fbytes[2];
        fread(fbytes, 4, 2, fp);
        // put into circular buffer
        for (j=0;j<10;j++)
            cbuf[cidx+(j<<12)] = (float)(bytes[j])/65536.f;
        cidx = (1+cidx)&4095;
        cnt++;
        cntstride++;
        if (fbytes[0]>54 && cnt>input_len && cntstride>stride)
        {
            cntstride = 0;
            train.y.vals[0][0] = (fbytes[0]-50)/200.0;
            for (k=0;k<net.w;k++)
            for (j=0;j<10;j++)
            {
                train.X.vals[0][(j*net.w)+k] = cbuf[(j<<12)+(((cidx-1-net.w+k)+4096)&4095)];
            }
            float *p = network_predict(net, train.X.vals[0]);
         //   fprintf(stderr, "%f, %f\n", train.y.vals[0][0], p[0]);
            p[0] = (p[0]*200.0)+50.0;
            fprintf(stdout, "%f, %f\n", fbytes[0], p[0]);
            fprintf(stderr, "%f, %f\n", fbytes[0], p[0]);
        }
    }
    fclose(fp);

    free_network(net);
    free_data(train);
}

void run_cgm(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train] [cfg] [train list] [test list]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [test] [cfg] [weights] [session bin]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    if(0==strcmp(argv[2], "train")) train_cgm(cfg, argv[4], argv[5]);
    else if(0==strcmp(argv[2], "test")) test_cgm(cfg, argv[4], argv[5]);

}


