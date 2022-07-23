#include "data_ops.h"

MNIST::MNIST(std::string filepath) {
    std::string paths = filepath + "\\train-images.idx3-ubyte";
    loadImage(paths.c_str(), "image"/*"image" or "label"*/, true);
    paths = filepath + "\\t10k-labels.idx1-ubyte";
    loadImage(paths.c_str(), "label"/*"image" or "label"*/, false);
    paths = filepath + "\\t10k-images.idx3-ubyte";
    loadImage(paths.c_str(), "image"/*"image" or "label"*/, false);
    paths = filepath + "\\train-labels.idx1-ubyte";
    loadImage(paths.c_str(), "label"/*"image" or "label"*/, true);
    printf_s("Init finish!\n");
}

int MNIST::ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNIST::loadImage(const char* fileName, const char* mode, bool isTrain) {
    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open()) {
        if (strcmp(mode, "image") == 0) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            unsigned char label;
            file.read((char*)&magic_number, sizeof(magic_number));
            file.read((char*)&number_of_images, sizeof(number_of_images));
            file.read((char*)&n_rows, sizeof(n_rows));
            file.read((char*)&n_cols, sizeof(n_cols));
            magic_number = ReverseInt(magic_number);
            number_of_images = ReverseInt(number_of_images);
            n_rows = ReverseInt(n_rows);
            n_cols = ReverseInt(n_cols);
            std::cout << "magic number = " << magic_number << std::endl;
            std::cout << "number of images = " << number_of_images << std::endl;
            std::cout << "rows = " << n_rows << std::endl;
            std::cout << "cols = " << n_cols << std::endl;
            for (int i = 0; i < number_of_images; i++)
            {
                vector<BYTE> tp;
                for (int r = 0; r < n_rows; r++)
                {
                    for (int c = 0; c < n_cols; c++)
                    {
                        unsigned char image = 0;
                        file.read((char*)&image, sizeof(image));
                        tp.push_back(image);
                    }
                }
                if (isTrain)train_images.push_back(tp);
                else test_images.push_back(tp);
            }
        }
        else {
            int magic_number = 0;
            int number_of_images = 0;
            file.read((char*)&magic_number, sizeof(magic_number));
            file.read((char*)&number_of_images, sizeof(number_of_images));
            magic_number = ReverseInt(magic_number);
            number_of_images = ReverseInt(number_of_images);
            std::cout << "magic number = " << magic_number << std::endl;
            std::cout << "number of images = " << number_of_images << std::endl;
            for (int i = 0; i < number_of_images; i++)
            {
                unsigned char label = 0;
                file.read((char*)&label, sizeof(label));
                if (isTrain)train_labels.push_back(label);
                else test_labels.push_back(label);
            }
        }
    }
}

tuple4 MNIST::image2vector() {
    return tuple4(train_images, train_labels, test_images, test_labels);
}