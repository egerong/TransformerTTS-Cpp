#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

//Eigen::MatrixXd vecToMat(std::vector<float> input, int nRows);
void matToCSV(Eigen::MatrixXd mat, std::string filePath);