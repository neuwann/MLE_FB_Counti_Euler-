#include <iostream>
#include <fstream>
#include <Eigen/Dense>

Eigen::MatrixXd readCSV(std::string file, int rows, int cols) {

  std::ifstream in(file);
  
  std::string line;

  int row = 0;
  int col = 0;

  Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 1; i < len; i++) {

        if (ptr[i] == ' ') {
          res(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      res(row, col) = atof(start);

      row++;
    }

    in.close();
  }
  return res;
}

int main(){
    Eigen::MatrixXd X=Eigen::MatrixXd(1000,3);
    X=readCSV("data.csv",1000,3);
    std::cout << "The matrix X is:\n" << X << "\n\n";
}