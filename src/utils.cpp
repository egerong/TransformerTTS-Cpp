#include "utils.h"

using namespace std;
using namespace Eigen;


MatrixXf vecToMat(vector<float> input, int nRows) {
    int nCols = input.size() / nRows;
    MatrixXf mat(nRows, nCols);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            mat(i, j) = input[i * nRows + j];
        }
    }
    return mat;
}

const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

void matToCSV(MatrixXf mat, string filePath) {
    ofstream file(filePath.c_str());
    file << mat.format(CSVFormat);
}