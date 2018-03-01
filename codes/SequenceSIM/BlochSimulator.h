/* 
BlochSimulator.h

Class for Bloch Simulations

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#pragma once

#include "ReferenceVolume.h"

#define Gamma 276.513

class BlochSimulator
{
private:

   //Input Variables
	double B0;     // T
    ReferenceVolume* referenceVolume;
	unsigned int numberOfPulseSamples;
	double* RFMagnitude; // w1 [T]
	double* RFPhase;     // rad
	double* XGradient;   // mT per meter
	double* YGradient;   // mT per meter
	double* Timesteps;   // s
	bool* ADC;           // true if sampling data
	bool isAllocated;

public:
	BlochSimulator();
	~BlochSimulator();
	void Initialize(double b0, ReferenceVolume* refVolume);
	void AllocateMemory(unsigned int numberOfSamples);
	void FreeMemory();
	void SetRFPulses(double* magnitude, double* phase);
	void SetRFPulses(unsigned int pos, double magnitude, double phase);
	void SetGradients(double* xGradient, double* yGradient);
	void SetGradients(unsigned int pos, double xGradient, double yGradient);
	void SetTimesteps(double* timeSteps);
	void SetTimesteps(unsigned int pos, double timeSteps);
	void SetADC(bool* adc);
	void SetADC(unsigned int pos, bool adc);
	void RunSimulation(MatrixXcd& kSpace);
	Vector3d SolveBlochEquation(Vector3d &M0, Matrix3d &A, Vector3d &C, double& t, int numApprox = 6);
};