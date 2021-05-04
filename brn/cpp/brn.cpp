// brn.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include "memory"
#include "recon.h"
#include "time.h"


int main()
{
    clock_t s, e;
    int resolution = 512;
    int nViews = 360;
    int nBins = 600;
    char filePath[128] = "E:\\BRNData\\val\\projData_False\\990.raw";
    float* result = new float[resolution * resolution];
    memset(result, 0, sizeof(float) * resolution * resolution);


    BRN brn;
    s = clock();
    clock_t s1, e1;
    int iter = 40;
    for (size_t i = 0; i < iter; i++)
    {
        s1 = clock();
        brn.recon(filePath, result);
        e1 = clock();
        std::cout << double(e1 - s1) / 1000 << std::endl;
    }
    e = clock();
    FILE* fp;
    fopen_s(&fp, "1.raw", "wb");
    fwrite(result, 1, sizeof(float) * resolution * resolution, fp);
    fclose(fp);

    
    std::cout << "avg time" << double(e - s)/1000/iter << std::endl;
    std::cout << "Hello World!\n";
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
