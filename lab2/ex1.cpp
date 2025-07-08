#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

const double Eh = 27.211;
const double l = 10 / 0.052918;
const double m = 0.067;
const double eps_r = 12.5;
const double tol = 1e-9;

double it_v(const VectorXd& XY, int i, int j, double eps_r, double l) {
    return 1.0 / (eps_r * sqrt(pow(XY[i] - XY[j], 2) + pow(l, 2)));
}

MatrixXd it_f(int m, double dx, const MatrixXd& psi, const VectorXd& XY, double eps_r, double l) {
    int n = psi.rows();
    MatrixXd f_out = MatrixXd::Zero(n, n);
    
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            f_out(i, j) = -(psi(i+1, j) + psi(i-1, j) + psi(i, j+1) + psi(i, j-1) - 4 * psi(i, j)) / (2 * m * dx * dx) 
                           + psi(i, j) * it_v(XY, i, j, eps_r, l);
        }
    }
    return f_out;
}

MatrixXd it_psi(const MatrixXd& psi, const MatrixXd& f, double dtau) {
    return psi - dtau * f;
}

MatrixXd it_norm(const MatrixXd& psi, double dx) {
    double c = (psi.array().square().sum()) * dx * dx;
    return psi / sqrt(c);
}

double it_e(const MatrixXd& psi, const MatrixXd& f, double dx) {
    return (psi.array() * f.array()).sum() * dx * dx;
}

bool it_test(double E0, double E1, double tol) {
    return abs((E1 - E0) / E1) < tol;
}

int main() {
    int n = 41;
    double a = 30 / 0.052918;
    double dx = 2 * a / (n - 1);
    VectorXd XY = VectorXd::LinSpaced(n, -a, a);
    double dtau = m * dx * dx * 0.4;

    MatrixXd psi_it = MatrixXd::Zero(n, n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            psi_it(i, j) = dist(gen);
        }
    }
    
    double E = 0, E_next;
    bool test = false;
    int iterations = 0;
    vector<double> Energies;

    while (!test) {
        MatrixXd f = it_f(m, dx, psi_it, XY, eps_r, l);
        MatrixXd psi_next = it_psi(psi_it, f, dtau);
        psi_next = it_norm(psi_next, dx);
        E_next = it_e(psi_next, f, dx);

        test = it_test(E, E_next, tol);
        if (!test) {
            psi_it = psi_next;
        }
        
        iterations++;
        E = E_next;
        if (iterations % 100 == 0) {
            Energies.push_back(E * Eh);
        }
    }

    cout << "Final Energy: " << E * Eh << " eV" << endl;
    return 0;
}
