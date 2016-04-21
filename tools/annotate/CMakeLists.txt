PROJECT(annotate)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2 -s -pipe -msse3 -std=c++0x")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR};/home/jcronje/anaconda2/include)
#;/usr/local/include;/usr/include/ffmpeg;/opt/local/include)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR};/opt/local/lib;/usr/local/lib;/usr/lib)

SET(SOURCES
  main.cpp
)

ADD_EXECUTABLE(annotate ${SOURCES})
TARGET_LINK_LIBRARIES(annotate pthread glog boost_system hdf5_hl hdf5 opencv_core opencv_highgui opencv_imgproc opencv_objdetect)
