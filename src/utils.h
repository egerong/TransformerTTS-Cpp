#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

Eigen::MatrixXf vecToMat(std::vector<float> input, int nRows);
void matToCSV(Eigen::MatrixXf mat, std::string filePath);