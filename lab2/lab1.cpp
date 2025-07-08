#include<iostream>
#include<cmath>
#include<random>

using namespace std;

const double epsilon_r = 12.5;
const double kappa = 1 / epsilon_r;
const int n = 41;
const double bohr_length = 0.052;
const double l = 10.0 / bohr_length;
const double a = 30.0 / bohr_length;
const double mass = 0.067;
vector<double> x;

class RandomNumberGenerator {
    private:
        random_device rd; 
        mt19937 gen;      
        uniform_real_distribution<> dis;
    public:
        RandomNumberGenerator() :  gen(rd()), dis(0.0, 1.0) {} ;

        double get_random() {
            return dis(gen);
        }
};

class Exercise1 {
    private:
        double dx = 2 * a / (n - 1);
        vector<double> x;
        vector<vector<double> > psi_func;

        RandomNumberGenerator rng;

        Exercise1() {
            x.resize(n);
            for (int i = 0; i < n; ++i) {
                x[i] = -a + i * dx;  
            }
        }

        double potential_int(double x1, double x2) {
            return kappa / sqrt((x1 - x2) * (x1 - x2) + l * l);
        }

        double hamiltonian_on_psi(vector<vector<double> > psi, int iterator_i, int iterator_j) {
            if (iterator_i <= 0 || iterator_i >= n - 1 || iterator_j <= 0 || iterator_j >= n - 1) {
                throw out_of_range("Index out of bounds in hamiltonian_on_psi");
            }

            double numerator = psi[iterator_i + 1][iterator_j] + psi[iterator_i - 1][iterator_j] +
                               psi[iterator_i][iterator_j + 1] + psi[iterator_i][iterator_j - 1] -
                               4 * psi[iterator_i][iterator_j];
            return -1 / (2 * mass) * (1 / (dx * dx)) * numerator +
                   potential_int(x[iterator_i], x[iterator_j]) * psi[iterator_i][iterator_j];
        }

};

int main() {
    return 0;
}
