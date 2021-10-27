#include "utils.h"

using namespace std;
using namespace Eigen;

const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

void matToCSV(MatrixXd mat, string filePath) {
    ofstream file(filePath.c_str());
    file << mat.format(CSVFormat);
}