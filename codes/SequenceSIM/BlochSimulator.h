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
	RELAX,
    DEPHASE
};


class BlochSimulator
{
private:

    ReferenceVolume* referenceVolume; // ref volume with t1,t2 and pd
	MatrixXd Mx, My, Mz;              // magnetization in x,y and z
	double kx, ky;					  // x and y gradient moments
	uint32_t numKSpaceSamples, numSpins;

public:
	//Constructor & Destructor
	BlochSimulator(){};
	~BlochSimulator(){};

	uint32_t PixelPositionToIdx(uint32_t row, uint32_t col)
	{
		return row + col * referenceVolume->GetNumberOfRows();
	}

	// return Samples
	uint32_t GetNumberOfKSpaceSamples(){ return numKSpaceSamples; };
	
	// Init field strength and volume
	void Initialize(ReferenceVolume* refVolume, uint32_t numSamples, uint32_t numberOfSpins);

	// Simulation functions
	void RunSimulation(ExternalSequence& sequence, std::vector<KSpaceEvent>& kSpace);

	//Update A matrix
	void SetRFPulse(Matrix3d& A, double rfAmplitude, double rfPhase, double rfFreqOffset);
	void SetOffresonance(Matrix3d& A, double dw);

	// Apply event that is the same in the entire volume e.g. relaxation phase, non-selective pulse
	void ApplyGlobalEventToVolume(SeqBlock* seqBlock);
	void ApplyEventToVolume(SeqBlock* seqBlock);
	void AcquireKSpaceLine(std::vector<KSpaceEvent>& kSpace, SeqBlock* seqBlock, uint32_t &currentADC);

	// appy bloch simulation to pixel
	void ApplyBlochSimulationPixel(uint32_t row, uint32_t col, uint32_t spin, Matrix3d& A, double t, BlochSolverType type);
    
};