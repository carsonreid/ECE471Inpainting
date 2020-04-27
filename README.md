# ECE471 Inpainting

An Inpainting Computer Vision project

## Prerequisites

Todo

* MacOS Users: X11 if not already installed

## Running the Algorithm

1. Compile the C++ project using g++ inside the "cimg_inpaint" folder:

    Windows (MingW): 
    `` g++ -o mini_inpaint mini_inpaint.cpp -O2 -lgdi32 ``

    MacOS:
    `` g++ -o mini_inpaint mini_inpaint.cpp -O2 -lm -lpthread -I/usr/X11R6/include -L/usr/X11R6/lib -lm -lpthread -lX11 ``

