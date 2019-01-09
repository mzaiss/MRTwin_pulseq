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


Vector3d Precess(Vector3d &Mi, Matrix3d &A, double t)
{
	double b, bx, by, bz, nx, ny, nz, c, s, k;
	bx = -A(2, 1)*t;
	by = -A(2, 0)*t;
	bz = A(0, 1)*t;
	b = sqrt(bx*bx + by * by + bz * bz);
	Vector3d M(Mi.x(), Mi.y(), Mi.z());
	if (b > 0.0)
	{
		bx /= b;  nx = Mi.x();
		by /= b;  ny = Mi.y();
		bz /= b;  nz = Mi.z();

		c = sin(0.5*b); c = 2.0*c*c;
		s = sin(b);
		k = nx * bx + ny * by + nz * bz;

		M(0) += (bx*k - nx)*c + (ny*bz - nz * by)*s;
		M(1) += (by*k - ny)*c + (nz*bx - nx * bz)*s;
		M(2) += (bz*k - nz)*c + (nx*by - ny * bx)*s;
	}
	return M;
}

Vector3d Relax(Vector3d &Mi, Matrix3d &A, double M0, double t)
{
	Vector3d M(Mi.x(), Mi.y(), Mi.z());
	double e2 = exp(A(0, 0)*t);
	double e1 = exp(A(2, 2)*t);
	M(0) *= e2;
	M(1) *= e2;
	M(2) = M0 + e1 * (Mi.z() - M0);
	return M;
}

Vector3d Dephase(Vector3d &Mi, Matrix3d &A, double t)
{
	double b,bz, nx, ny, nz, c, s, k;
	bz = A(0, 1)*t;
	b = fabs(bz);
	Vector3d M(Mi.x(), Mi.y(), Mi.z());
	if (b > 0.0)
	{
		nx = Mi.x();
        ny = Mi.y();
		nz = Mi.z();
		bz /= b;

		c = sin(0.5*b); c = 2.0*c*c;
		s = sin(b);
		k = nz * bz;

		M(0) += (-nx)*c + ny*bz*s;
		M(1) += (-ny)*c + (-nx*bz)*s;
	}
	return M;
}



