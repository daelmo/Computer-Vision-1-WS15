#-------------------------------------------------
#
# Project created by QtCreator 2015-10-05T13:34:42
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = Task2
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

LIBS += `pkg-config opencv --libs`
SOURCES += main.cpp

# LIBS += -L/usr/local/lib
unix:!macx: LIBS += -lopencv_core -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_video -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts
