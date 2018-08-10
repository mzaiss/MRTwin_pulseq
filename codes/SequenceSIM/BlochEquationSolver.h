/*
Solver function for 1 pool bloch equations
MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#pragma once

#include "Eigen\eigen"
using namespace Eigen;

/*

This function solves the Bloch equation
M = (Mi + A^-1 * C) * exp(A*t) - A^-1 * C
The matrix exponent is calculated with the Pade approximation
see: Golub and Van Loan, Matrix Computations, Algorithm 11.3-1.

Input:  Mi: 3x1 init magnetization vector
        A : 3x3 matrix with 
			|-R2           dw0          -w1*sin(phi) |
		A = |-dw0         -R2           -w1*cos(phi) |
			| w1*sin(phi)  w1*cos(phi)  -R1          |
        C : 3x1 vector with the relaxation parameters
        t : duration of the simulation [s]
        numApprox: number of approximations for pade method

  Output: Vector3d: the resulting new 3x1 magnetization vector

*/
Vector3d SolveBlochEquation(Vector3d &Mi, Matrix3d &A, Vector3d &C, double t, int numApprox = 6)
{
	Vector3d AInvT = A.inverse()*C; // helper variable A^-1 * C
	Matrix3d At = A*t;				// helper variable A * t	
	//solve exponential with pade method
	int infExp; //infinity exponent of the matrix
	int j;
	std::frexp(At.lpNorm<Infinity>(), &infExp); // pade method is only stable if ||A||inf / 2^j <= 0.5
	j = std::max(0, infExp + 1);
	At = At*(1.0 / (pow(2, j)));
	//the algorithm usually starts with D = X = N = Identity and c = 1
	// since c is alway 0.5 after the first loop, we can start in the second round and init the matrices corresponding to that 	
	Matrix3d X(At); // X = A after first loop
	double c = 0.5; // c = 0.5 after first loop
	Matrix3d N = Matrix3d::Identity() + c*At;
	Matrix3d D = Matrix3d::Identity() - c*At;
	bool p = true; // D +- cX is dependent from (-1)^k, fastest way is with changing boolean in the loop
	double q = numApprox;
	Matrix3d cX; // helper variable for c * X
	// run the approximation 
	for (int k = 2; k <= q; k++)
	{
		c = c * (q - k + 1) / (k*(2 * q - k + 1));
		X = At*X;
		cX = c*X;
		N = N + cX;
		if (p)
			D = D + cX;
		else
			D = D - cX;
		p = !p;
	}
	Matrix3d F = D.inverse()*N; // solve D*F = N for F
	for (int k = 1; k <= j; k++)
	{
		F *= F;
	}
	return F*(Mi + AInvT) - AInvT; // return the result
}