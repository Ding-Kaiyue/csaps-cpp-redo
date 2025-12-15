#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>
#include "src/csaps.h"

using json = nlohmann::json;
using namespace csaps;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.json> <output.json>\n";
        return 1;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "Cannot open: " << argv[1] << "\n";
        return 1;
    }

    json test_data;
    input_file >> test_data;
    input_file.close();

    json results;
    results["test_results"] = json::object();

    auto& test_cases = test_data["test_cases"];

    for (const auto& test : test_cases) {
        std::string name = test["name"];

        try {
            auto x_vec = test["x"].get<std::vector<double>>();
            DoubleArray x = Eigen::Map<DoubleArray>(x_vec.data(), x_vec.size());

            json result_obj;
            result_obj["success"] = false;

            // Univariate case
            if (test.contains("y") && !test["y"].is_array() || (test["y"].is_array() && !test["y"][0].is_array())) {
                auto y_vec = test["y"].get<std::vector<double>>();
                DoubleArray y = Eigen::Map<DoubleArray>(y_vec.data(), y_vec.size());

                double smooth = -1.0;
                if (test.contains("smooth") && !test["smooth"].is_null()) {
                    smooth = test["smooth"].get<double>();
                }

                bool normalizedsmooth = false;
                if (test.contains("normalizedsmooth") && !test["normalizedsmooth"].is_null()) {
                    normalizedsmooth = test["normalizedsmooth"].get<bool>();
                }

                DoubleArray weights;
                if (test.contains("weights") && !test["weights"].is_null()) {
                    auto w_vec = test["weights"].get<std::vector<double>>();
                    weights = Eigen::Map<DoubleArray>(w_vec.data(), w_vec.size());
                }

                UnivariateCubicSmoothingSpline spline(x, y, weights, smooth, normalizedsmooth);

                result_obj["smooth"] = spline.GetSmooth();
                result_obj["success"] = true;

                // Evaluate
                auto xi_vec = test["xi"].get<std::vector<double>>();
                DoubleArray xi = Eigen::Map<DoubleArray>(xi_vec.data(), xi_vec.size());

                int nu = test.value("nu", 0);
                bool extrapolate = test.value("extrapolate", true);

                DoubleArray yi = spline(xi, nu, extrapolate);

                std::vector<double> y_result(yi.data(), yi.data() + yi.size());
                result_obj["y"] = y_result;
            }
            // Multivariate case
            else {
                auto y_data = test["y"].get<std::vector<std::vector<double>>>();
                size_t n_dims = y_data.size();
                size_t n_points = y_data[0].size();

                // Python format is (n_dims, n_points), C++ expects (n_points, n_dims)
                DoubleArray2D y(n_points, n_dims);
                for (size_t i = 0; i < n_dims; i++) {
                    for (size_t j = 0; j < n_points; j++) {
                        y(j, i) = y_data[i][j];  // Transpose
                    }
                }

                double smooth = -1.0;
                if (test.contains("smooth") && !test["smooth"].is_null()) {
                    smooth = test["smooth"].get<double>();
                }
                MultivariateCubicSmoothingSpline spline(x, y, DoubleArray(), smooth);

                auto smooths = spline.GetSmooths();
                result_obj["smooth"] = smooths(0);  // Use first dimension's smooth
                result_obj["success"] = true;

                // Evaluate
                auto xi_vec = test["xi"].get<std::vector<double>>();
                DoubleArray xi = Eigen::Map<DoubleArray>(xi_vec.data(), xi_vec.size());

                DoubleArray2D yi = spline(xi);
                // yi is (n_query, n_dims), convert to Python format (n_dims, n_query)
                std::vector<std::vector<double>> y_result;
                for (size_t d = 0; d < yi.cols(); d++) {
                    std::vector<double> col(yi.col(d).data(), yi.col(d).data() + yi.rows());
                    y_result.push_back(col);
                }
                result_obj["y"] = y_result;
            }

            results["test_results"][name] = result_obj;
        } catch (const std::exception& e) {
            results["test_results"][name]["success"] = false;
            results["test_results"][name]["error"] = std::string(e.what());
        }
    }

    std::ofstream output_file(argv[2]);
    output_file << std::setw(2) << results << "\n";
    output_file.close();

    return 0;
}
