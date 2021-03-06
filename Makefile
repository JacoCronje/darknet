GPU=1
OPENCV=1
DEBUG=0

ARCH= --gpu-architecture=compute_20 --gpu-code=compute_20 

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread -lstdc++ 
COMMON= 
#CFLAGS=-Wall -Wfatal-errors
CFLAGS=-Wfatal-errors

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
#COMMON+= `pkg-config --cflags opencv`
COMMON+= -I/home/jcronje/anaconda2/include/
#LDFLAGS+= -L/home/jcronje/anaconda2/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc
#LDFLAGS+= `pkg-config --libs opencv`
LDFLAGS+= -lopencv_core -lopencv_highgui -lopencv_imgproc
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o keypoint_layer.o mnist.o convolutional_layer.o driver.o list.o image.o activations.o im2col.o augment_layer.o shrinkadd_layer.o shrinkmax_layer.o sidebyside_layer.o compact_layer.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o imagenet.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o rnn.o rnn_vid.o crnn_layer.o coco_demo.o tag.o cifar.o yolo_demo.o go.o stl10.o
ifeq ($(GPU), 1) 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o shrinkmax_layer_kernels.o softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o augment_layer_kernels.o keypoint_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

