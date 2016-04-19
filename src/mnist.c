#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

void train_mnist(char *cfgfile, char *weightfile)
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
    int N = 60000;

    data train;
    data test;
    train = load_mnist_data("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte", 60000);
    test = load_mnist_data("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", 10000);

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
            fprintf(stderr, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fprintf(stdout, "%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
            fflush(stdout);
            time=clock();
        }
        if (isnan(loss) || isnan(avg_loss))
        {
            // NaN detected!!!
            fprintf(stderr, "NaN!\n");

            free_network(net);
            load_weights(&net, backup_net);
            nanCount++;
            if (nanCount>=5) break;
            continue;
        }
        if(get_current_batch(net)%1000 == 0){
            float *acc = network_accuracies(net, test, 2);
            a[2] = acc[0];
            a[3] = acc[1];
            float mse1 = acc[2];
            float *accT = network_accuracies(net, train, 2);
            a[0] = accT[0];
            a[1] = accT[1];
            float mse2 = accT[2];
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
}

void train_mnist_distill(char *cfgfile, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);

    char *backup_directory = "backup";
    int classes = 10;
    int N = 50000;

    int epoch = (*net.seen)/N;

    data train;// = load_all_mnist10();
    matrix soft = csv_to_matrix("results/ensemble.csv");

    float weight = .9;
    scale_matrix(soft, weight);
    scale_matrix(train.y, 1. - weight);
    matrix_add_matrix(soft, train.y);

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        clock_t time=clock();

        float loss = train_network_sgd(net, train, 1);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.95 + loss*.05;
        if(get_current_batch(net)%100 == 0)
        {
            printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        }
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free(base);
    free_data(train);
}

void test_mnist_multi(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));

    float avg_acc = 0;
    data test;
    test = load_mnist_data("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", 10000);

    int i;
    for(i = 0; i < test.X.rows; ++i){
        image im = float_to_image(96, 96, 3, test.X.vals[i]);

        float pred[10] = {0};

        float *p = network_predict(net, im.data);
        axpy_cpu(10, 1, p, 1, pred, 1);
        flip_image(im);
        p = network_predict(net, im.data);
        axpy_cpu(10, 1, p, 1, pred, 1);

        int index = max_index(pred, 10);
        int class = max_index(test.y.vals[i], 10);
        if(index == class) avg_acc += 1;
        free_image(im);
    }
    printf("%4d: %.2f%%\n", i, 100.*avg_acc/(i+1));
}

void test_mnist(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    clock_t time;
    float avg_acc = 0;
    float avg_top5 = 0;

    data test;
    test = load_mnist_data("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", 10000);

    time=clock();

    float *acc = network_accuracies(net, test, 2);
    avg_acc += acc[0];
    avg_top5 += acc[1];
    printf("top1: %f, %lf seconds, %d images\n", avg_acc, sec(clock()-time), test.X.rows);

    save_network_feature_maps(net, 0, net.n-3, "network", 10, 2);

    free_data(test);
}

void test_mnist_csv(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    data test;
    test = load_mnist_data("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", 10000);

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

void test_mnist_csvtrain(char *filename, char *weightfile)
{
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    data test;
    test = load_mnist_data("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", 10000);

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

void eval_mnist_csv()
{
    data test;
    test = load_mnist_data("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte", 10000);

    matrix pred = csv_to_matrix("results/combined.csv");
    fprintf(stderr, "%d %d\n", pred.rows, pred.cols);

    fprintf(stderr, "Accuracy: %f\n", matrix_topk_accuracy(test.y, pred, 1));
    free_data(test);
    free_matrix(pred);
}


void run_mnist(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")) train_mnist(cfg, weights);
    else if(0==strcmp(argv[2], "distill")) train_mnist_distill(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_mnist(cfg, weights);
    else if(0==strcmp(argv[2], "multi")) test_mnist_multi(cfg, weights);
    else if(0==strcmp(argv[2], "csv")) test_mnist_csv(cfg, weights);
    else if(0==strcmp(argv[2], "csvtrain")) test_mnist_csvtrain(cfg, weights);
    else if(0==strcmp(argv[2], "eval")) eval_mnist_csv();
}


