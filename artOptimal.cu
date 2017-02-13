/*
@author: lugao
*/

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
#include <sstream>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/random.h>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;

static chrono::time_point <chrono::high_resolution_clock> tstart;

static void tic() {
    tstart = chrono::high_resolution_clock::now();
}

static void toc() {
    auto tstop = chrono::high_resolution_clock::now();
    auto dt = 1.e-9 * std::chrono::duration_cast<chrono::nanoseconds>(tstop - tstart).count();
    printf("Elapsed time is %f seconds.\n", dt);
}


void cumsum2D(int *m, int v, int w, int h, int *out) {
    int W = w + 1;
    int H = h + 1;

    for (int x = 0; x < W; ++x)
        out[0 * W + x] = 0;
    for (int y = 0; y < H; ++y)
        out[y * W + 0] = 0;
    for (int y = 1; y < H; y++) {
        for (int x = 1; x < W; ++x) {
            out[y * W + x] = m[(y - 1) * w + (x - 1)] == v;
        }
    }
    for (int y = 0; y < H; y++) {
        for (int x = 1; x < W; ++x) {
            out[y * W + x] = out[y * W + x] + out[y * W + (x - 1)];
        }
    }
    for (int x = 0; x < W; ++x) {
        for (int y = 1; y < H; y++) {
            out[y * W + x] = out[(y - 1) * W + x] + out[y * W + x];
        }
    }
}
template<typename Numeric>
void show(Numeric *image, int w, int h, const string &map = "Greys") {
    plt::figure();
    plt::imshow(image, w, h, map);
    plt::xlim(0, w);
    plt::ylim(0, h);
    plt::show();
}


__global__
void paintSquare(int *a, int w, int h, int x, int y, int s, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s * s) {
        int xx = i % s;
        int yy = i / s;
        a[(y + yy) * w + (x + xx)] = v;
    }
}

int appendOperations(vector <string> &operations, int *image, int w, int h, int x, int y, int s) {
    int painted = 0;
    stringstream ss;
    for (int xx = x; xx < x + s; ++xx) {
        for (int yy = y; yy < y + s; ++yy) {
            if (image[yy * w + xx] == 0) {
                ss << "ERASE," << xx << "," << yy;
                operations.push_back(ss.str());
                ss.str(string());
            } else if (image[yy * w + xx] == 1) {
                painted++;
            }
        }
    }
    if (painted != 0) {
        ss << "FILL," << x << "," << y << "," << s;
        operations.push_back(ss.str());
    }
    return painted;
}

__global__
void buildScoreMatrix(int *m0, int *m1, /*int *m2,*/ int w, int h, float *score, bool onlyFill) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) {
        int x = i % w;
        int y = i / w;
        int minD = min(w, h);
        float maxScore = -1;
        int maxS = 0;
        int maxCount0 = 0;
        //int maxCount1 = 0;

        for (int s = minD; s > 0; --s) {
            if (x + s <= w && y + s <= h) {
                int count0 = m0[y * (w + 1) + x] +
                             m0[(y + s) * (w + 1) + (x + s)] -
                             m0[(y + s) * (w + 1) + x] -
                             m0[y * (w + 1) + (x + s)];

                int count1 = m1[y * (w + 1) + x] +
                             m1[(y + s) * (w + 1) + (x + s)] -
                             m1[(y + s) * (w + 1) + x] -
                             m1[y * (w + 1) + (x + s)];
                float score = count1 / (1.0f + count0 * 0.390f);
                if (onlyFill)
                    score = (1.0f * count1) / (1.0f + 10 * count0);
                if (maxScore < score ||
                    (maxScore == score && count0 < maxCount0)) {
                    maxScore = score;
                    maxS = s;
                    maxCount0 = count0;
                    //maxCount1 = count1;
                }
            }
        }
        score[y * w + x] = maxScore - 0.1f * maxCount0;//(1.0f * maxCount1) / (1.0f + 0.390f*maxCount0);
        if (onlyFill)
            score[y * w + x] = maxScore;
        score[(w * h) + y * w + x] = maxS;
    }
}


void greedySolution(int pixelsLeft, int *image, int w, int h, vector <string> &operations, int debugLevel = 1) {
    /* Debug levels
     * 0: no report nor image
     * 1: report every 1 second
     * 2: report and image every 1 second
     * 3: report and image every step
     */
    int totalPixels = pixelsLeft;
    int prediction = pixelsLeft;

    int *image_d, *m0_d, *m1_d; //, *m2_d
    float *score_d;
    int *m0 = (int *) malloc((w + 1) * (h + 1) * sizeof(int));
    int *m1 = (int *) malloc((w + 1) * (h + 1) * sizeof(int));
    float *score = (float *) malloc(2 * w * h * sizeof(float));

    cudaMalloc(&m0_d, (w + 1) * (h + 1) * sizeof(int));
    cudaMalloc(&m1_d, (w + 1) * (h + 1) * sizeof(int));
    cudaMalloc(&image_d, w * h * sizeof(int));
    cudaMalloc(&score_d, 2 * w * h * sizeof(float));

    tic();
    int minPred = w * h;
    int counter = 0;
    auto timer = chrono::high_resolution_clock::now();
    while (pixelsLeft > 0) {
        //tic();

        cumsum2D(image, 0, w, h, m0);
        cumsum2D(image, 1, w, h, m1);

        cudaMemcpy(m0_d, m0, (w + 1) * (h + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(m1_d, m1, (w + 1) * (h + 1) * sizeof(int), cudaMemcpyHostToDevice);

        //-----------------------------------CUDA-----------------------------------
        buildScoreMatrix << < (w * h + 1023) / 1024, 1024 >> >
                                                     (m0_d, m1_d, w, h, score_d, prediction <= 3050);
        cudaMemcpy(score, score_d, 2 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
        //-----------------------------------CUDA-----------------------------------

        thrust::host_vector<float> h_vec(score, score + w * h);
        thrust::device_vector<float> d_vec = h_vec;

        thrust::device_vector<float>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());

        unsigned int position = iter - d_vec.begin();
        float max_val = *iter;

        int sqX = position % w;
        int sqY = position / w;
        int sqS = (int) score[w * h + position];
        if (debugLevel >= 3)
            printf("Best square at (%d,%d) with size %d\n", sqX, sqY, sqS);

        pixelsLeft -= appendOperations(operations, image, w, h, sqX, sqY, sqS);
        cudaMemcpy(image_d, image, w * h * sizeof(int), cudaMemcpyHostToDevice);

        //-----------------------------------CUDA-----------------------------------
        paintSquare << < (sqS * sqS + 1023) / 1024, 1024 >> > (image_d, w, h, sqX, sqY, sqS, 2);
        cudaMemcpy(image, image_d, w * h * sizeof(int), cudaMemcpyDeviceToHost);
        //-----------------------------------CUDA-----------------------------------

        h_vec.clear();
        thrust::host_vector<float>().swap(h_vec);

        d_vec.clear();
        thrust::device_vector<float>().swap(d_vec);

        int totalOperations = (int) operations.size();
        prediction = totalOperations + pixelsLeft;

        minPred = min(prediction, minPred);
        auto now = chrono::high_resolution_clock::now();
        auto dt = 1.e-9 * std::chrono::duration_cast<chrono::nanoseconds>(now - timer).count();
        if (pixelsLeft <= 0 ||
            (counter % 10 == 0 && prediction < 3500) ||
            (debugLevel <= 2 && dt > 1)) {
            printf("[%6.2f%%] Total operations: %4d | Pixels left: %5d | Prediction <= %5d | MinPred = %5d \n",
                   (totalPixels - pixelsLeft) * 100.0f / totalPixels,
                   totalOperations, pixelsLeft, prediction, minPred);
            if (debugLevel == 2)
                show(image, w, h);

            if (debugLevel <= 2)
                timer = chrono::high_resolution_clock::now();
        }
        counter++;
        if (debugLevel >= 3)
            show(image, w, h);
    }

    toc();

    free(m0);
    free(m1);
    free(image);
    free(score);

    cudaFree(m0_d);
    cudaFree(m1_d);
    cudaFree(image_d);
    cudaFree(score_d);
}

int readMatrix(int *a, int w, int h) {
    int totalPixels = 0;
    for (int y = 0; y < h; y++) {
        char buffer[w + 10];
        if (scanf("%s", buffer) == 0)
            printf("Failed to read line %d", y);

        for (int x = 0; x < w; ++x) {
            a[y * w + x] = buffer[x] == '#';
            totalPixels += buffer[x] == '#';
        }
    }
    return totalPixels;
}

int main(void) {
    //Redirect input file to stdin
    if (freopen("../input_0.txt", "r", stdin) == nullptr)
        printf("Failed redirecting input file to stdin.");

    //Reade the matrix dimensions
    int w, h;
    if (scanf("%d,%d", &w, &h) != 2)
        printf("Failed reading matrix dimentions.");

    //Allocate image matrix and operations vector
    int *image = (int *) malloc(w * h * sizeof(int));
    vector <string> operations;

    //Read the matrix from the file
    int pixelsLeft = readMatrix(image, w, h);

    //Execute the greedy solution
    greedySolution(pixelsLeft, image, w, h, operations);


    if(system("python ../verificationTool.py") != 0)
        printf("Verification failed!");
    //Write the solution to the file in a reversed order
    printf("Writting solution...");

    if (freopen("../output_0.txt", "w", stdout) == nullptr)
        printf("Failed redirecting stdout to output file.");

    for (auto it = operations.rbegin(); it < operations.rend(); it++)
        printf("%s\n", it->c_str());
    fclose(stdout);

}