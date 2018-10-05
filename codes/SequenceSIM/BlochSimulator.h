/* 
BlochSimulator.h

Class for Bloch Simulations

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#pragma once

#include "ReferenceVolume.h"
#include "ExternalSequence.h"

#define Gamma 42.577

struct KSpaceEvent
{
	std::complex<double> kSample = std::complex<double>(0.0, 0.0);
	double kX = 0.0;
	double kY = 0.0;
};

enum BlochSolverType 
{
	FULL,
	PRECESS,
	RELAX
};


class BlochSimulator
{
private:

    ReferenceVolume* referenceVolume; // ref volume with t1,t2 and pd
	MatrixXd Mx, My, Mz;              // magnetization in x,y and z
	double kx, ky;					  // x and y gradient moments
	unsigned int numKSpaceSamples;

public:
	//Constructor & Destructor
	BlochSimulator(){};
	~BlochSimulator(){};

	// return Samples
	unsigned int GetNumberOfKSpaceSamples(){ return numKSpaceSamples; };
	
	// Init field strength and volume
	void Initialize(ReferenceVolume* refVolume, unsigned int numSamples);

	// Simulation functions
	void RunSimulation(ExternalSequence& sequence, std::vector<KSpaceEvent>& kSpace);

	//Update A matrix
	void SetRFPulse(Matrix3d& A, double rfAmplitude, double rfPhase);
	void SetOffresonance(Matrix3d& A, double dw);

	// Apply event that is the same in the entire volume e.g. relaxation phase, non-selective pulse
	void ApplyGlobalEventToVolume(SeqBlock* seqBlock);
	void ApplyEventToVolume(SeqBlock* seqBlock);
	void AcquireKSpaceLine(std::vector<KSpaceEvent>& kSpace, SeqBlock* seqBlock, unsigned int &currentADC);

	// appy bloch simulation to pixel
	void ApplyBlochSimulationPixel(unsigned int row, unsigned int col, Matrix3d& A, double t, BlochSolverType type);
};