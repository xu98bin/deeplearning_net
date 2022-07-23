#ifndef DATA_OPS_H
#define DATA_OPS_H

#include<fstream>
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<filesystem>
#include<cstring>
#include<iostream>
#include<fstream>

using std::vector;
using std::fstream;
#define BYTE unsigned char
#define tuple4 std::tuple<vector<vector<BYTE>>, vector<BYTE>, vector<vector<BYTE>>, vector<BYTE>>

class MNIST {
public:
    vector<vector<BYTE>> train_images, test_images;
    vector<BYTE> train_labels, test_labels;
public:
    MNIST(std::string);

    int ReverseInt(int i);

    void loadImage(const char*, const char*, bool);

    tuple4 image2vector();
};

#endif