#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sine.h"
#include "cosine.h"


// reference: [1]Liu JinKun. Robot Control System Design and MATLAB Simulation[M]. Tsinghua University Press, 2008.
// [2]Lee T H, Harris C J. Adaptive neural network control of robotic manipulators[M]. World Scientific, 1998.

// global variables declaration
#define PI 3.14159
#define H 5  // hidden layer neurons number
#define ARRAY_SIZE  6000  // hidden layer neurons number

static double Ts = 0.005; // sampling period
static double t0 = 0.0; // start time
static double t1 = 30.0; // end time
static double c_D[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};  // inertia matrix RBF function center
static double c_G[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};  // gravity matrix RBF function center
static double c_C[2][5] = {{-1.0, -0.5, 0.0, 0.5, 1.0}, {-2.0, -1.0, 0.0, 1.0, 2.0}};  // Coriolis matrix RBF function center
static double b = 1.5;    // RBF function center width
static double m = 0.020;  // manipulator mass
static double g = 9.8;    // gravitational acceleration
static double l = 0.05;   // connecting rod length

double phi_D[H], phi_G[H], phi_C[H];
double weight_D[H], weight_G[H], weight_C[H];
double derivative_weight_D[H], derivative_weight_G[H], derivative_weight_C[H];

void saveArchiveToTxt(double *archive, int size, const char *filename) {
    FILE *file = fopen(filename, "w");
    
    if (file != NULL) {
        for (int i = 0; i < size; i++) {
            fprintf(file, "%lf\n", archive[i]);
        }
        fclose(file);
        printf("Archive saved to %s\n", filename);
    } else {
        printf("Failed to open file %s\n", filename);
    }

}

struct _archive{
    double q_archive[ARRAY_SIZE];
    double dq_archive[ARRAY_SIZE];
    double error_archive[ARRAY_SIZE];
    double error_velocity_archive[ARRAY_SIZE];
    double tol_archive[ARRAY_SIZE];
    double C_estimate_norm_archive[ARRAY_SIZE];
    double C_norm_archive[ARRAY_SIZE];
} archive;

Data qd, ddqd;
Data1 dqd;

struct Amp {
    double qd1;
    double dqd1;
    double ddqd1;
};

struct M0 {
    double qd2;
    double dqd2;
    double ddqd2;
};

struct B0 {
    double qd3;
    double dqd3;
    double ddqd3;
};

void mdlinput(Data *qd, Data1 *dqd, Data *ddqd, double Ts, double t0, double t1) {

    struct Amp amp;  // amplitude
    amp.qd1 = 0.5;
    amp.dqd1 = 0.5 * PI;
    amp.ddqd1 = -0.5 * pow(PI, 2);

    struct M0 m0;  // angular frequency
    m0.qd2 = PI;
    m0.dqd2 = PI;
    m0.ddqd2 = PI;

    struct B0 b0;  // vertical shift
    b0.qd3 = 0.0;
    b0.dqd3 = 0.0;
    b0.ddqd3 = 0.0;

    sine(qd, Ts, t0, t1, amp.qd1, m0.qd2, b0.qd3);         // desired angular displacement
    cosine(dqd, Ts, t0, t1, amp.dqd1, m0.dqd2, b0.dqd3);   // desired angular velocity
    sine(ddqd, Ts, t0, t1, amp.ddqd1, m0.ddqd2, b0.ddqd3); // desired angular acceleration

}

struct _system_state{
    double q;    // actual angular displacement
    double dq;   // actual angular velocity
    double ddq;  // actual angular acceleration
} system_state;

struct _torque{
    double tol;  // control torque
    double tol_m; // Eq. 3.56 define, control law based on model estimation
    double tol_r; // Eq. 3.57 define, robust term for network modeling error
} torque;

struct _dynamics{
    double D0;   // inertia matrix for nominal model
    double G0;   // gravity matrix for nominal model
    double C0;   // Coriolis matrix for nominal model
    double D_norm;   // two-paradigm number of D0
    double G_norm;   // two-paradigm number of G0
    double C_norm;   // two-paradigm number of C0
    double DSNN_estimate; // estimate of RBF network modeling term DSNN
    double GSNN_estimate; // estimate of RBF network modeling term GSNN
    double CDNN_estimate; // estimate of RBF network modeling term CDNN
    double C_estimate_norm; // second-paradigm number of estimate of RBF network modeling term CDNN
} dynamics;

struct _controller{
    double controller_u1;
    double controller_u2;
    double controller_u3;
    double controller_u4;
    double controller_u5;
    double controller_out1;
    double controller_out2;
    double err;            // angular displacement error
    double err_velocity;   // angular velocity error
    double Lambda_D;       // Eq. 3.60 define
    double Lambda_G;       // Eq. 3.62 define
    double Lambda_C;       // Eq. 3.61 define
    double Lambda;         // error's weight factor
    double r;              // Eq. 3.48 define
    double dqr;            // derivative of r
    double ddqr;           // second-order derivative of r
    double integral;       // integral term
    double Kr;             // r term factor
    double Kp;             // proportionality factor
    double Ki;             // integral factor
} controller;

void CONTROL_init(){
    system_state.q = 0.5;
    system_state.dq = 0.0;
    controller.controller_u1 = qd.y[0];
    controller.controller_u2 = dqd.y[0];
    controller.controller_u3 = ddqd.y[0];
    controller.controller_u4 = system_state.q;
    controller.controller_u5 = system_state.dq;
    controller.err = qd.y[0] - system_state.q;
    controller.err_velocity = dqd.y[0] - system_state.dq;
    controller.Lambda_D = 3;
    controller.Lambda_G = 6;
    controller.Lambda_C = 6;
    controller.Lambda = 5;
    controller.r = controller.err_velocity + controller.Lambda * controller.err; // Eq. 3.48
    controller.integral = 0.0;
    controller.Kr = 0.10;
    controller.Kp = 5;
    controller.Ki = 1.0;
}

struct _plant{
    double plant_u1;
    double plant_u2;
    double plant_out1;
    double plant_out2;
    double plant_out3;   
} plant;

void PLANT_init(){
    system_state.q = 0.5;
    system_state.dq = 0.0;
    plant.plant_u1 = 0.0;
    plant.plant_out1 = system_state.q;
    plant.plant_out2 = system_state.dq;
    plant.plant_out3 = 0.0;
}

double PLANT_realize(int i){
    plant.plant_u1 = torque.tol;
    dynamics.D0 = 0.1 + 0.06 * sin(system_state.q);
    dynamics.C0 = 0.03 * cos(system_state.q);
    dynamics.G0 = m * g * l * cos(system_state.q);
    // printf("dynamics.G0 = %f\n", dynamics.G0);

    system_state.ddq = (1.0/dynamics.D0) * (-dynamics.C0 * system_state.dq - dynamics.G0 + torque.tol);
    system_state.dq = system_state.dq + system_state.ddq * Ts;
    system_state.q = system_state.q + system_state.dq * Ts;
    
    dynamics.D_norm = dynamics.D0;
    dynamics.C_norm = dynamics.C0;
    dynamics.G_norm = dynamics.G0;

    plant.plant_out1 = system_state.q;
    plant.plant_out2 = system_state.dq;
    plant.plant_out3 = dynamics.C_norm;
    archive.C_norm_archive[i] = dynamics.C_norm;
}

double CONTROL_realize(int i){
    controller.controller_u1 = qd.y[i];
    controller.controller_u2 = dqd.y[i];
    controller.controller_u3 = ddqd.y[i];

    controller.controller_u4 = system_state.q;
    controller.controller_u5 = system_state.dq;
    // printf("system_state.dq = %f\n", system_state.dq);
    archive.q_archive[i] = system_state.q;
    archive.dq_archive[i] = system_state.dq;

    for (int j = 0; j < H; j++) {
        phi_D[j] = exp(-pow(system_state.q - c_D[j], 2) / (b * b));  // output of the inertia matrix RBF function
    }
    for (int j = 0; j < H; j++) {
        phi_G[j] = exp(-pow(system_state.q - c_G[j], 2) / (b * b));  // output of the gravity matrix RBF function
    }
    for (int j = 0; j < H; j++) {
        phi_C[j] = exp((-pow(system_state.q - c_C[0][j], 2) - pow(system_state.dq - c_C[1][j], 2))/ (b * b));  // output of the Coriolis matrix RBF function
        // printf("phi_C[%d] = %f\n", j, phi_C[j]);
    }

    for (int j = 0; j < H; j++) {
        weight_D[j] = 0.3;  // inertia matrix RBF network weight
        weight_G[j] = 0.3;  // gravity matrix RBF network weight
        weight_C[j] = 0.3;  // Coriolis matrix RBF network weight
    }

    controller.err = qd.y[i] - system_state.q;
    archive.error_archive[i] = controller.err;
    controller.err_velocity = dqd.y[i] - system_state.dq;
    // printf("controller.err_velocity = %f\n", controller.err_velocity);
    archive.error_velocity_archive[i] = controller.err_velocity;
    controller.r = controller.err_velocity + controller.Lambda * controller.err;
    controller.dqr = dqd.y[i] + controller.Lambda * controller.err;
    controller.ddqr = ddqd.y[i] + controller.Lambda * controller.err_velocity;
    // printf("controller.ddqr = %f\n", controller.ddqr);

    // adaptive law
    for (int j = 0; j < H; j++) {
        derivative_weight_D[j] = controller.Lambda_D * phi_D[j] * controller.ddqr * controller.r;  // Eq. 3.60
    }
    for (int j = 0; j < H; j++) {
        derivative_weight_G[j] = controller.Lambda_G * phi_G[j] * controller.r;  // Eq. 3.62
    }
    for (int j = 0; j < H; j++) {
        derivative_weight_C[j] = controller.Lambda_C * phi_C[j] * controller.dqr * controller.r;  // Eq. 3.61
        // printf("derivative_weight_C[%d] = %f\n", j, derivative_weight_C[j]);
    }

    for (int j = 0; j < H; j++) {
        weight_D[j] = weight_D[j] + derivative_weight_D[j] * Ts;
    }
    for (int j = 0; j < H; j++) {
        weight_G[j] = weight_G[j] + derivative_weight_G[j] * Ts;
    }
    for (int j = 0; j < H; j++) {
        weight_C[j] = weight_C[j] + derivative_weight_C[j] * Ts;
    }

    controller.integral += controller.r;

    for (int j = 0; j < H; j++) {
        dynamics.DSNN_estimate = weight_D[j] * phi_D[j];
    }
    for (int j = 0; j < H; j++) {
        dynamics.GSNN_estimate = weight_G[j] * phi_G[j];
    }
    for (int j = 0; j < H; j++) {
        dynamics.CDNN_estimate = weight_C[j] * phi_C[j];
    }

    dynamics.C_estimate_norm = dynamics.CDNN_estimate;
    torque.tol_m = dynamics.DSNN_estimate * controller.ddqr + dynamics.CDNN_estimate * controller.dqr + dynamics.GSNN_estimate; // Eq. 3.56, control law based on model estimation

    if (controller.r >= 0)
        torque.tol_r = controller.Kr;
    else
        torque.tol_r = - controller.Kr;

    torque.tol = torque.tol_m + controller.Kp * controller.r + controller.Ki * controller.integral + torque.tol_r; // Eq. 3.55, control law
    archive.tol_archive[i] = torque.tol;
    controller.controller_out1 = torque.tol;
    controller.controller_out2 = dynamics.C_estimate_norm;
    archive.C_estimate_norm_archive[i] = dynamics.C_estimate_norm;
}

int main(){


    mdlinput(&qd, &dqd, &ddqd, Ts, t0, t1); // system input

    CONTROL_init();  // initialize controller parameter
    PLANT_init();  // initialize plant parameter

    for (int i = 0; i < ARRAY_SIZE; i++) {
        double time = i * Ts + t0;
        printf("time at step %d: %f\n", i, time);
        CONTROL_realize(i);
        PLANT_realize(i);
    }

    saveArchiveToTxt(qd.y, ARRAY_SIZE, "qd_y.txt");
    saveArchiveToTxt(archive.q_archive, ARRAY_SIZE, "q_archive.txt");
    saveArchiveToTxt(archive.dq_archive, ARRAY_SIZE, "dq_archive.txt");
    saveArchiveToTxt(archive.error_archive, ARRAY_SIZE, "error_archive.txt");
    saveArchiveToTxt(archive.error_velocity_archive, ARRAY_SIZE, "error_velocity_archive.txt");
    saveArchiveToTxt(archive.tol_archive, ARRAY_SIZE, "tol_archive.txt");
    saveArchiveToTxt(archive.C_estimate_norm_archive, ARRAY_SIZE, "C_estimate_norm_archive.txt");
    saveArchiveToTxt(archive.C_norm_archive, ARRAY_SIZE, "C_norm_archive.txt");

    return 0;
}
