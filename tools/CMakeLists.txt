PROJECT(preparePlot)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#SET(CMAKE_CXX_COMPILER "g++")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2 -s -pipe -msse3 -std=c++0x")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR};/usr/local/include;/opt/local/include;)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR};/opt/local/lib)

SET(SOURCES
  preparePlot.cpp
)

ADD_EXECUTABLE(preparePlot ${SOURCES})
TARGET_LINK_LIBRARIES(preparePlot)
