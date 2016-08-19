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
        free_image(img);
    }

}

void train_driver(char *cfgfile, char *trainlist, char *weightfile)
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

    data train, test;

    fprintf(stderr, "Loading training labels\n");

    int i,j,k;
    FILE *fp = fopen("data/driver_imgs_list.csv", "r");
    if(!fp) file_error("data/driver_imgs_list.csv");
    char tmp[512];
    int *lbl = malloc(65536*2*4);
    memset(lbl, 0xff, 65536*2*4);
    int *lblperson = malloc(65536*2*4);
    memset(lblperson, 0xff, 65536*2*4);
    int pbit[100];
    int pcnt[100];
    memset(pbit, 0, sizeof(pbit));
    memset(pcnt, 0, sizeof(pcnt));
    fscanf(fp, "%s", tmp);
    int N = 0;
    while (fscanf(fp, "%s", tmp)==1)
    {
        int cl = tmp[6]-'0';
        int num = tmp[12]-'0';
        int person = (tmp[1]-'0')*100 + (tmp[2]-'0')*10 + (tmp[3]-'0');

        i = 13;
        while (tmp[i]!='.')
        {
            num = num*10 + (tmp[i]-'0');
            i++;
        }
        ++N;
        lbl[num] = cl;
        lblperson[num] = person;
        pbit[person] = 1;
        pcnt[person]++;
//        fprintf(stderr, "%d: %d %d\n", count, cl, num);
    }
    fclose(fp);

    int personlist[100];
    int nperson = 0;
    for (i=0;i<100;i++)
    {
        if (pbit[i]==1)
        {
            personlist[nperson] = i;
            nperson++;
        }
    }
    int Ntrain = N;
    int Ntest = 0;
    for (j=0;j<5;j++)
    {
        do
        {
            i = rand()%nperson;
        } while (pbit[personlist[i]]==0);
        Ntrain -= pcnt[personlist[i]];
        Ntest  += pcnt[personlist[i]];
        pbit[personlist[i]] = 0;
    }
    fprintf(stderr, "Training images = %d Testing images = %d\n", Ntrain, Ntest);

//N=1000;
    train.shallow = 0;
    train.X = make_matrix(Ntrain, net.w*net.h*net.c);
    train.y = make_matrix(Ntrain, 10);
    test.shallow = 0;
    test.X = make_matrix(Ntest, net.w*net.h*net.c);
    test.y = make_matrix(Ntest, 10);
    int itrain = 0;
    int itest = 0;

    fprintf(stderr, "Loading images\n");
    list *plist = get_paths(trainlist);//"data/driver_imgs128.txt");
    char **paths = (char **)list_to_array(plist);
    for (i=0;i<plist->size && i<N;i++)
    {
        if ((i%100)==0)
            fprintf(stderr, "progress = %d of %d\n", i, plist->size);
        char *path = paths[i];
        image img = load_image(path, net.w, net.h, net.c);
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

                if (pbit[lblperson[num]]==1)
                {
                    // train set
                    train.y.vals[itrain][(int)lbl[num]] = 1;
                    for(j = 0; j < train.X.cols; ++j){
                        train.X.vals[itrain][j] = (float)(img.data[j]);//*255.f - 110.f;
                    }
                    itrain++;
                } else
                {
                    // test set
                    test.y.vals[itest][(int)lbl[num]] = 1;
                    for(j = 0; j < test.X.cols; ++j){
                        test.X.vals[itest][j] = (float)(img.data[j]);//*255.f - 110.f;
                    }
                    itest++;
                }
                break;
            }
        free_image(img);
    }
    //scale_data_rows(train, 1./255);

    clock_t time=clock();
    float a[4];

    char backup_net[256];
    int nanCount = 0;

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        if(get_current_batch(net)%100 == 0)
        {
            fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(N*0.9), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(N*0.9), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fflush(stdout);
            time=clock();
        }

      //  save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);
       // return 0;

        if (isnan(loss) || isnan(avg_loss))
        {
            // NaN detected!!!
            free_network(net);
            load_weights(&net, backup_net);
            nanCount++;
            if (nanCount>=5) break;
            continue;
        }
        if(get_current_batch(net)%500 == 0){
            float *acc = network_accuracies(net, test, 2);
            a[2] = acc[0];
            a[3] = acc[1];
            float mse2 = acc[2];
            float *accT = network_accuracies(net, train, 2);
            a[0] = accT[0];
            a[1] = accT[1];
            float mse1 = accT[2];
            fprintf(stderr, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse1, a[2], a[3], mse2);
            fprintf(stdout, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse1, a[2], a[3], mse2);
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
    free_data(test);
}

void trainfold_driver(char *cfgfile, char *trainlist, int K, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "backup";
    network net;// = parse_network_cfg(cfgfile);
 //   if(weightfile){
   //     load_weights(&net, weightfile);
   // }

    {
        char buff[256];
        sprintf(buff, "%s/%s.txt", backup_directory, base);
        freopen(buff, "w", stdout);
    }

   // fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    int i,j,k;
    int foldK;

    data train, test;

    fprintf(stderr, "Loading training labels\n");

    FILE *fp = fopen("data/driver_imgs_list.csv", "r");
    if(!fp) file_error("data/driver_imgs_list.csv");
    char tmp[512];
    int *lbl = malloc(65536*2*4);
    memset(lbl, 0xff, 65536*2*4);
    int *lblperson = malloc(65536*2*4);
    memset(lblperson, 0xff, 65536*2*4);
    int pbit[128*32];
    int pcnt[128];
    memset(pbit, 0, sizeof(pbit));
    memset(pcnt, 0, sizeof(pcnt));
    fscanf(fp, "%s", tmp);
    int N = 0;
    while (fscanf(fp, "%s", tmp)==1)
    {
        int cl = tmp[6]-'0';
        int num = tmp[12]-'0';
        int person = (tmp[1]-'0')*100 + (tmp[2]-'0')*10 + (tmp[3]-'0');

        i = 13;
        while (tmp[i]!='.')
        {
            num = num*10 + (tmp[i]-'0');
            i++;
        }
        ++N;
        lbl[num] = cl;
        lblperson[num] = person;
        for (k=0;k<K;k++)
            pbit[person+(k<<7)] = 1;
        pcnt[person]++;
//        fprintf(stderr, "%d: %d %d\n", count, cl, num);
    }
    fclose(fp);

    int personlist[100];
    int nperson = 0;
    for (i=0;i<100;i++)
    {
        if (pbit[i]==1)
        {
            personlist[nperson] = i;
            nperson++;
        }
    }
    int Ntrain[32] = {N};
    int Ntest[32] = {0};
    srand(123579L);
    for (k=0;k<K;k++)
    {
        Ntrain[k] = N;
        Ntest[k] = 0;
        for (j=0;j<5;j++)
        {
            do
            {
                i = rand()%nperson;
            } while (pbit[personlist[i]+(k<<7)]==0);
            Ntrain[k] -= pcnt[personlist[i]];
            Ntest[k]  += pcnt[personlist[i]];
            pbit[personlist[i]+(k<<7)] = 0;
        }
    }
    //free_network(net);

    data_seed = time(0);
    srand(time(0));

    double validError = 0;
    for (foldK=0;foldK<K;foldK++)
    {
        net = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&net, weightfile);
        }
        fprintf(stderr, "Fold %d : Training images = %d Testing images = %d\n", foldK, Ntrain[foldK], Ntest[foldK]);
        train.shallow = 0;
        train.X = make_matrix(Ntrain[foldK], net.w*net.h*net.c);
        train.y = make_matrix(Ntrain[foldK], 10);
        test.shallow = 0;
        test.X = make_matrix(Ntest[foldK], net.w*net.h*net.c);
        test.y = make_matrix(Ntest[foldK], 10);
        int itrain = 0;
        int itest = 0;

        fprintf(stderr, "Loading images\n");
        list *plist = get_paths(trainlist);//"data/driver_imgs128.txt");
        char **paths = (char **)list_to_array(plist);
        for (i=0;i<plist->size && i<N;i++)
        {
            if ((i%2000)==0)
                fprintf(stderr, "progress = %d of %d\n", i, plist->size);
            char *path = paths[i];
            image img = load_image(path, net.w, net.h, net.c);
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
                    if (lbl[num]<0 || lbl[num]>9)
                        fprintf(stderr, "ERROR = %d  %d %d\n", lbl[num], num, i);

                    if (pbit[lblperson[num]+(foldK<<7)]==1)
                    {
                        // train set
                        train.y.vals[itrain][(int)lbl[num]] = 1;
                        for(j = 0; j < train.X.cols; ++j){
                            train.X.vals[itrain][j] = (float)(img.data[j]);//*255.f - 110.f;
                        }
                        itrain++;
                    } else
                    {
                        // test set
                        test.y.vals[itest][(int)lbl[num]] = 1;
                        for(j = 0; j < test.X.cols; ++j){
                            test.X.vals[itest][j] = (float)(img.data[j]);//*255.f - 110.f;
                        }
                        itest++;
                    }
                    break;
                }
            free_image(img);
        }

        clock_t time=clock();
        float a[4];

        char backup_net[256];
        int nanCount = 0;

        while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

            float loss = train_network_sgd(net, train, 1);
            if(avg_loss == -1) avg_loss = loss;
            avg_loss = avg_loss*.95 + loss*.05;
            if(get_current_batch(net)%100 == 0)
            {
                fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(Ntrain[foldK]), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
                fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(Ntrain[foldK]), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
                fflush(stdout);
                time=clock();
            }
          //  save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);
           // return 0;
            if (isnan(loss) || isnan(avg_loss))
            {
                // NaN detected!!!
                free_network(net);
                load_weights(&net, backup_net);
                nanCount++;
                if (nanCount>=5) break;
                continue;
            }
            if(get_current_batch(net)%500 == 0){
                float *acc = network_accuracies(net, test, 2);
                a[2] = acc[0];
                a[3] = acc[1];
                float mse2 = acc[2];
                float *accT = network_accuracies(net, train, 2);
                a[0] = accT[0];
                a[1] = accT[1];
                float mse1 = accT[2];
                fprintf(stderr, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse1, a[2], a[3], mse2);
                fprintf(stdout, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse1, a[2], a[3], mse2);
                fflush(stdout);
                char buff[256];
                sprintf(buff, "%s/%s_k%d_%d.weights",backup_directory,base, foldK, get_current_batch(net));
                sprintf(backup_net, "%s/%s_k%d_%d.weights",backup_directory,base, foldK, get_current_batch(net));
                save_weights(net, buff);
                nanCount = 0;
                if (get_current_batch(net)+1>=net.max_batches)
                {
                    validError += mse2;
                }
            }
        }
        fprintf(stderr, "Validation error %f\n", validError/(foldK+1));
        fprintf(stdout, "Validation error %f\n", validError/(foldK+1));
        char buff[256];
        sprintf(buff, "%s/%s_k%d.weights", backup_directory, base, foldK);
        save_weights(net, buff);

        free_network(net);
        free_data(train);
        free_data(test);
    }
}

void trainparts_fold_driver(char *cfgfile, char *trainlist, int K, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));


    int augmentParts = 1;
    if (K<0)
    {
        K = -K;
        augmentParts = 0;
    }

    float avg_loss = -1;
    char base[1024];
    if (augmentParts==1)
    {
        sprintf(base, "%sAUG",basecfg(cfgfile));
    } else
    {
        sprintf(base, "%sNO",basecfg(cfgfile));
    }
    //char *base = basecfg(cfgfile);
    char *backup_directory = "backup";
    network net;
    {
        char buff[256];
        sprintf(buff, "%s/%s.txt", backup_directory, base);
        freopen(buff, "w", stdout);
    }

    int i,j,k,z,jj;
    int foldK;

    data train, test;

    fprintf(stderr, "Loading training labels\n");
    FILE *fp = fopen("data/driver_imgs_list.csv", "r");
    if(!fp) file_error("data/driver_imgs_list.csv");
    char tmp[512];
    int *lbl = malloc(65536*2*4);
    memset(lbl, 0xff, 65536*2*4);
    int *lblperson = malloc(65536*2*4);
    memset(lblperson, 0xff, 65536*2*4);
    int pbit[128*32];
    int pcnt[128];
    memset(pbit, 0, sizeof(pbit));
    memset(pcnt, 0, sizeof(pcnt));
    fscanf(fp, "%s", tmp);
    int N = 0;
    while (fscanf(fp, "%s", tmp)==1)
    {
        int cl = tmp[6]-'0';
        int num = tmp[12]-'0';
        int person = (tmp[1]-'0')*100 + (tmp[2]-'0')*10 + (tmp[3]-'0');
        i = 13;
        while (tmp[i]!='.')
        {
            num = num*10 + (tmp[i]-'0');
            i++;
        }
        ++N;
        lbl[num] = cl;
        lblperson[num] = person;
        for (k=0;k<K;k++)
            pbit[person+(k<<7)] = 1;
        pcnt[person]++;
    }
    fclose(fp);

    int personlist[100];
    int nperson = 0;
    for (i=0;i<100;i++)
    {
        if (pbit[i]==1)
        {
            personlist[nperson] = i;
            nperson++;
        }
    }
    int Ntrain[32] = {N};
    int Ntest[32] = {0};
    srand(123579L);
    for (k=0;k<K;k++)
    {
        Ntrain[k] = N;
        Ntest[k] = 0;
        for (j=0;j<5;j++)
        {
            do
            {
                i = rand()%nperson;
            } while (pbit[personlist[i]+(k<<7)]==0);
            Ntrain[k] -= pcnt[personlist[i]];
            Ntest[k]  += pcnt[personlist[i]];
            pbit[personlist[i]+(k<<7)] = 0;
        }
    }

    data_seed = time(0);
    srand(time(0));

    float *X;//= calloc(net.batch*net.w*net.h*net.c, sizeof(float));
    float *y;// = calloc(net.batch*10, sizeof(float));

    double validError = 0;
    for (foldK=0;foldK<K;foldK++)
    {
        net = parse_network_cfg(cfgfile);
        if (foldK==0)
        {
            X = calloc(net.batch*net.w*net.h*net.c, sizeof(float));
            y = calloc(net.batch*10, sizeof(float));
        }
        if (weightfile)
        {
            load_weights(&net, weightfile);
        }
        ///-----------------
       // Ntrain[foldK] = 4756;
      //  Ntest[foldK] = 909;
        //Ntest[foldK] = 62*64;

        fprintf(stderr, "Fold %d : Training images = %d Testing images = %d\n", foldK, Ntrain[foldK], Ntest[foldK]);
        fprintf(stderr, "net.w=%d net.h=%d net.c=%d\n", net.w, net.h, net.c);
   //     train.shallow = 0;
    //    train.X = make_matrix(net.batch, net.w*net.h*net.c);
     //   train.y = make_matrix(net.batch, 10);
//        train.X = make_matrix(Ntrain[foldK], net.w*net.h*net.c);
//        train.y = make_matrix(Ntrain[foldK], 10);
        test.shallow = 0;
        test.X = make_matrix(Ntest[foldK], net.w*net.h*net.c);
        test.y = make_matrix(Ntest[foldK], 10);
        int itrain = 0;
        int itest = 0;
        char *pathclass[10][4096];
        int pathj[10][4096];
        int nimgs[10];
        for (i=0;i<10;i++) nimgs[i] = 0;

        fprintf(stderr, "Loading images\n");
        list *plist = get_paths(trainlist);//"data/driver_imgs128.txt");
        char **paths = (char **)list_to_array(plist);
        for (i=0;i<plist->size && i<N;i++)
        {
            if ((i%2000)==0)
                fprintf(stderr, "progress = %d of %d\n", i, plist->size);
            if ((i%500)==0)
            {
                fprintf(stderr, "itest=%d\n", itest);
                for (j=0;j<10;j++)
                    fprintf(stderr, "%d ", nimgs[j]);
                fprintf(stderr, "\n");
            }
            char *path = paths[i];
          //  fprintf(stderr, "%s\n", path);
         //   image img = load_image(path, net.w, net.h, net.c);
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
                    if (lbl[num]<0 || lbl[num]>9)
                        fprintf(stderr, "ERROR = %d  %d %d\n", lbl[num], num, i);

                    if (pbit[lblperson[num]+(foldK<<7)]==1)
                    {
                        int cl = (int)lbl[num];
                        pathj[cl][nimgs[cl]] = j-3;
                        pathclass[cl][nimgs[cl]] = path;
                        nimgs[cl]++;
                        /*
                        train.y.vals[itrain][(int)lbl[num]] = 1;
                        for (z=0;z<4;z++)
                        {
                            path[j-3] = '0'+z;
                            image img = load_image(path, net.w, net.h, 3);
                            for(jj = 0; jj< net.w*net.h*3; ++jj)
                            {
                                train.X.vals[itrain][jj+net.w*net.h*3*z] = (float)(img.data[jj]);
                            }
                            free_image(img);
                        }
                        itrain++;
                        */
                    } else if (itest<Ntest[foldK])
                    {
                        int jk;
                        test.y.vals[itest][(int)lbl[num]] = 1;
                        for (z=0;z<4;z++)
                        {
                            path[j-3] = '0'+z;
                          //  fprintf(stderr, "%s\n", path);
                            image img = load_image(path, net.w, net.h, 3);
                            for(jk = 0; jk < net.w*net.h*3; ++jk)
                            {
                                test.X.vals[itest][jk+(net.w*net.h*3*(z))] = (float)(img.data[jk]);
                            }
                            free_image(img);
                        }
                        itest++;
                    }
                    break;
                }

        }

        clock_t time=clock();
        float a[4];

        char backup_net[256];
        int nanCount = 0;

        fprintf(stderr, "itest=%d\n", itest);
        for (i=0;i<10;i++)
            fprintf(stderr, "%d ", nimgs[i]);
        fprintf(stderr, "\n");
        while(get_current_batch(net) < net.max_batches || net.max_batches == 0){

            // TODO : random shuffle internals
            itrain = 0;
            for (i=0;i<net.batch;i++)
            {
                int cl = rand()%10;
                //for (j=0;j<10;j++) train.y.vals[itrain][j] = 0;
                //train.y.vals[itrain][cl] = 1;
                for (j=0;j<10;j++) y[i*10+j] = 0;
                y[i*10+cl] = 1;
                int iimg;
                if (!augmentParts)
                {
                    iimg = rand()%nimgs[cl];
                }
                for (z=0;z<4;z++)
                {
                    if (augmentParts)
                    {
                        iimg = rand()%nimgs[cl]; /// move this outside loop not to mix class parts
                    }
                //    fprintf(stderr, "%d %d %s %d\n", i, z, pathclass[cl][iimg], pathj[cl][iimg]);
                    pathclass[cl][iimg][pathj[cl][iimg]] = '0'+z;
                    image img = load_image(pathclass[cl][iimg], net.w, net.h, 3);
                  //  fprintf(stderr, "%d %d %d\n", img.w, img.h, img.c);
                   // fprintf(stderr, "%d %d %s\n", i, z, pathclass[cl][iimg]);
                    for(jj = 0; jj< net.w*net.h*3; ++jj)
                    {
                        //train.X.vals[itrain][jj+net.w*net.h*3*z] = (float)(img.data[jj]);
                       X[(i*net.w*net.h*net.c)+jj+(net.w*net.h*3*z)] = (float)(img.data[jj]);
                    }
                    free_image(img);
                }
                itrain++;
            }
//fprintf(stderr, "pre-train\n");
            float loss = train_network_datum(net, X, y) / net.batch;
//fprintf(stderr, "post-train\n");
            //float loss = train_network_sgd(net, train, 1);
            if(avg_loss == -1) avg_loss = loss;
            avg_loss = avg_loss*.95 + loss*.05;
            if(get_current_batch(net)%100== 0)
            {
                fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(Ntrain[foldK]), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
                fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/(Ntrain[foldK]), loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
                fflush(stdout);
                time=clock();
            }
       //   save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);
       //   return 0;
            if (isnan(loss) || isnan(avg_loss))
            {
                // NaN detected!!!
                free_network(net);
                load_weights(&net, backup_net);
                nanCount++;
                if (nanCount>=5) break;
                continue;
            }
            if(get_current_batch(net)%500 == 0){
//                for (i=0;i<test.X.rows;i++)
//                for (z=0;z<test.X.cols;z++)
//                test.X.vals[i][z] = 0;//(float)(img.data[jk]);

                float *acc = network_accuracies(net, test, 2);
//                   save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);
//                    return 0;
                a[2] = acc[0];
                a[3] = acc[1];
                float mse2 = acc[2];
               // float *accT = network_accuracies(net, train, 2);
                a[0] = 0;//accT[0];
                a[1] = 0;//accT[1];
                float mse1 = 0;//accT[2];
                fprintf(stderr, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse1, a[2], a[3], mse2);
                fprintf(stdout, "Accuracy: train(%f %f %f) test(%f %f %f)\n", a[0], a[1], mse1, a[2], a[3], mse2);
                fflush(stdout);
                char buff[256];
                sprintf(buff, "%s/%s_k%d_%d.weights",backup_directory,base, foldK, get_current_batch(net));
                sprintf(backup_net, "%s/%s_k%d_%d.weights",backup_directory,base, foldK, get_current_batch(net));
                save_weights(net, buff);
                nanCount = 0;
                if (get_current_batch(net)+1>=net.max_batches)
                {
                    validError += mse2;
                }
            }
        }
        fprintf(stderr, "Validation error %f\n", validError/(foldK+1));
        fprintf(stdout, "Validation error %f\n", validError/(foldK+1));
        char buff[256];
        sprintf(buff, "%s/%s_k%d.weights", backup_directory, base, foldK);
        save_weights(net, buff);

        free_network(net);
       // free_data(train);
        free_data(test);
    }
}

void extractparts_driver(char *cfgfile, char *weightfile, char *imglist, char *folder)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));
   // freopen(outfile, "w", stdout);

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
        for (j=strlen(paths[i]);j>=0;j--)
        {
            if (paths[i][j]=='/')
            {
                j++;
                k = 0;
                while (paths[i][j]!=0)
                {
                    tmp[k++] = paths[i][j];
                    j++;
                }
                tmp[k-4] = 0;
                break;
            }
        }
       // sscanf(paths[i],"%[^,]%s\n",tmp2,tmp);
      //  sprintf(tmp, "%s/%s", folder, tmp2);
        image img = load_image_color(paths[i], net.w, net.h);
        for (j = 0; j < net.w*net.h*3; ++j)
        {
            dat[j+nidx*net.w*net.h*3] = ((double)img.data[j]);// / 255;
        }
        float *p = network_predict_gpu(net, dat);
        image imgF = load_image_color(paths[i], 640, 480);

        for (k=0;k<4;k++)
        {
            int ix = 640*p[k*2];
            int iy = 480*p[k*2+1];
            if (ix<0) ix = 0;
            if (ix>639) ix = 639;
            if (iy<0) iy = 0;
            if (iy>639) iy = 639;
            image cropF2 = crop_image(imgF, ix-128, iy-128, 256, 256);
            image cropF = resize_image(cropF2, 128, 128);
            char buff[256];
            tmp[0] = '0'+k;
            sprintf(buff, "%s/%s", folder, tmp);
            save_image(cropF, buff);
            free_image(cropF);
            free_image(cropF2);
        }
        free_image(img);
        free_image(imgF);
    }
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
            dat[j+nidx*net.w*net.h*3] = ((double)img.data[j]);// / 255;

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

    fprintf(stderr, "Loading training data\n");

    char tmp[1512];
    char tmp2[1512];
    int i,j,k;
    FILE *fp = fopen(imglist, "r");
    if(!fp) file_error(imglist);
    // read header
  //  fscanf(fp, "%s\n", tmp);
    train.shallow = 0;
    matrix X = make_matrix(880, net.w*net.h*3);
    matrix y = make_matrix(880, 2*4);
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
            //fprintf(stderr, "Loading [%s]\n", tmp2);
            image img = load_image_color(tmp2, net.w, net.h);
            for(j = 0; j < X.cols; ++j){
                X.vals[YSZ][j] = (double)img.data[j];
            }
            free_image(img);
            YSZ++;
        }
    }
    //return;
  //  X = resize_matrix(X, YSZ);
  //  y = resize_matrix(y, YSZ);
    //train.X = X;
   // train.y = y;
  //  scale_data_rows(train, 1./255);
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

void test_driver(char *filename, char *weightfile, char *listfile)
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

    float *dat = malloc(net.w*net.h*net.c*16*sizeof(float));
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
        image img = load_image(path, net.w, net.h, net.c);
        for(j = 0; j < net.w*net.h*net.c; ++j){
            dat[j+nidx*net.w*net.h*net.c] = ((double)img.data[j]);// / 255;
            //dat[j+nidx*net.w*net.h*net.c] = (float)(img.data[j])*255.f - 110.f;
        }
       // printf("predicting\n");
        nidx++;
        if (nidx==1)
        {
            float *p = network_predict_gpu(net, dat);
            int kidx = 0;
            for (k=0;k<nidx;k++)
            {
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
            fprintf(stdout, "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", fname[k],
                    p[0+kidx],p[1+kidx],p[2+kidx],p[3+kidx],p[4+kidx],p[5+kidx],p[6+kidx],p[7+kidx],p[8+kidx],p[9+kidx] );
            kidx+=10;
        }
    }
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
        fprintf(stderr, "usage: %s %s [test] [cfg] [weights] [imagelist]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [trainpoints] [cfg] [imagefolder] [imagelist] [weights]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [testpoints] [cfg] [weights] [imagefolder] [imagelist] [outfile]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [train] [cfg] [imagelist] [weights]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [trainfold] [cfg] [imagelist] [k] [weights]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [extractparts] [cfg] [weights] [imagelist] [outfolder]\n", argv[0], argv[1]);
        fprintf(stderr, "usage: %s %s [trainpartsfold] [cfg] [imagelist] [k] [weights]\n", argv[0], argv[1]);
        return;
    }

    int w,h;
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train"))
    {
        train_driver(cfg, argv[4],  (argc>5 ? argv[5] : 0));
    }
    else if(0==strcmp(argv[2], "trainfold"))
    {
        trainfold_driver(cfg, argv[4], atoi(argv[5]), (argc>6 ? argv[6] : 0));
    }
    else if(0==strcmp(argv[2], "test"))
    {
        char *listfile = argv[5];
        test_driver(cfg, weights, listfile);
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
    else if(0==strcmp(argv[2], "extractparts"))
    {
        extractparts_driver(cfg, weights, argv[5], argv[6]);
    }
    else if(0==strcmp(argv[2], "trainpartsfold"))
    {
        trainparts_fold_driver(cfg, argv[4], atoi(argv[5]), (argc>6 ? argv[6] : 0));
    }


}


