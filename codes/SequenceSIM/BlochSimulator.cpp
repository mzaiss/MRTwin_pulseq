/*
BlochSimulator.cpp

Class for Bloch Simulations

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#include "BlochSimulator.h"


BlochSimulator::BlochSimulator() 
{
	isAllocated = false;
}

BlochSimulator::~BlochSimulator()
{
	FreeMemory();
}

void BlochSimulator::Initialize(double b0, ReferenceVolume* refVolume)
{
	B0 = b0;
	referenceVolume = refVolume;
}

void BlochSimulator::AllocateMemory(unsigned int numberOfSamples)
{
	if (isAllocated) {
		FreeMemory();
	}

	numberOfPulseSamples = numberOfSamples;
	RFMagnitude = new double[numberOfPulseSamples];
	RFPhase     = new double[numberOfPulseSamples];
	XGradient   = new double[numberOfPulseSamples];
	YGradient   = new double[numberOfPulseSamples];
	Timesteps   = new double[numberOfPulseSamples];
	ADC         = new bool[numberOfPulseSamples];
	isAllocated = true;
}

void BlochSimulator::FreeMemory()
{
	if (isAllocated)
	{
		delete[] RFMagnitude;
		delete[] RFPhase;
		delete[] XGradient;
		delete[] YGradient;
		delete[] Timesteps;
		delete[] ADC;
	}
}

void BlochSimulator::SetRFPulses(double* magnitude, double* phase)
{
	RFMagnitude = magnitude;
	RFPhase = phase;
}

void BlochSimulator::SetRFPulses(unsigned int pos, double magnitude, double phase)
{
	if(isAllocated) {
		RFMagnitude[pos] = magnitude;
		RFPhase[pos] = phase;
	}
}

void BlochSimulator::SetGradients(double* xGradient, double* yGradient)
{
	XGradient = xGradient;
	YGradient = yGradient;
}

void BlochSimulator::SetGradients(unsigned int pos, double xGradient, double yGradient)
{
	if (isAllocated) {
		XGradient[pos] = xGradient;
		YGradient[pos] = yGradient;
	}
}

void BlochSimulator::SetTimesteps(double* timeSteps)
{
	Timesteps = timeSteps;
}

void BlochSimulator::SetTimesteps(unsigned int pos, double timeStep)
{
	if (isAllocated) {
		Timesteps[pos] = timeStep;
	}
}

void BlochSimulator::SetADC(bool* adc)
{
	ADC = adc;
}

void BlochSimulator::SetADC(unsigned int pos, bool adc)
{
	if (isAllocated) {
		ADC[pos] = adc;
	}
}

void BlochSimulator::RunSimulation(MatrixXcd& kSpace)
{
	//start with liner reordereing fron left bottom to top right corner for now
	unsigned int kx = 0;
	unsigned int ky = 0;
	//init variables
	Matrix3d A = Matrix3d::Zero();
	double w1cp; // omega1 times cos phi
	double w1sp; // omega1 times sin phi

	unsigned int numRows = referenceVolume->GetNumberOfRows();
	unsigned int numCols = referenceVolume->GetNumberOfColumns();
	double pixelSize = referenceVolume->GetPixelSize();

	// for magnetization in the 3 direction we use matrices. makes it easier to calculate the sum later
	MatrixXd Mx = MatrixXd::Zero(numRows, numCols);
	MatrixXd My = Mx;
	MatrixXd Mz = referenceVolume->GetProtonDensityMap(); // init z mag with proton density

	// Matrix A for bloch eq
    //     |-R2           dw0          -w1*sin(phi) |
	// A = |-dw0         -R2           -w1*cos(phi) |
	//     | w1*sin(phi)  w1*cos(phi)  -R1          |
	for (unsigned int nSample = 0; nSample < numberOfPulseSamples; nSample++)
	{
		w1cp = RFMagnitude[nSample] * cos(RFPhase[nSample]);
		w1sp = RFMagnitude[nSample] * sin(RFPhase[nSample]);
		A(0, 2) = -w1sp;
		A(2, 0) =  w1sp;
		A(1, 2) = -w1cp;
		A(2, 1) =  w1cp;

		// we could do this two loops on the GPU as they are independent from each other
		for (unsigned int row = 0; row < numRows; row++) {
			// gradient: 10^-3, pixel size 10^-3, Gamma: 10^6 -> 1
			double yGrad = YGradient[nSample] * ((double)row / numRows) * pixelSize * Gamma;
			for (unsigned int col = 0; col < numCols; col++) {
				// get M0
				double M0 = referenceVolume->GetProtonDensityValue(row, col);
				if (M0  <= 0) { // skip if there is no tissue
					continue;
				}
				//prepare for bloch equation
				Vector3d Mi(Mx(row, col), My(row, col), Mz(row, col));
				double dw0 = XGradient[nSample] * ((double)col / numCols) * pixelSize * Gamma + yGrad;
				double R1 = 1.0 / referenceVolume->GetT1Value(row, col);
				double R2 = 1.0 / referenceVolume->GetT2Value(row, col);
				A(0, 0) = -R2;
				A(1, 1) = -R2;
				A(2, 2) = -R1;
				A(0, 1) = dw0;
				A(1, 0) = -dw0;
				Vector3d C(0.0, 0.0, M0*R1);
				Vector3d M = SolveBlochEquation(Mi, A, C, Timesteps[nSample]);
				Mx(row, col) = M.x();
				My(row, col) = M.y();
				Mz(row, col) = M.z();
			}
		}
		//get the signal if sampling event is true
		if (ADC[nSample])
		{
			kSpace(ky, kx++) = std::complex<double>(Mx.sum(), My.sum());
			if (kx == numCols)
			{
				kx = 0;
				ky++;
				if (ky == numRows)
				{
					// k-space is full, we can stop here
					return;
				}
			}
		}

	}
}


Vector3d BlochSimulator::SolveBlochEquation(Vector3d &M0, Matrix3d &A, Vector3d &C, double& t, int numApprox)
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
	return F*(M0 + AInvT) - AInvT; // return the result
}