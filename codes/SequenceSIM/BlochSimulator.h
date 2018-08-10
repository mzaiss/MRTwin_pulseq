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

class BlochSimulator
{
private:

    ReferenceVolume* referenceVolume; // ref volume with t1,t2 and pd
	MatrixXd Mx, My, Mz;              // magnetization in x,y and z

public:
	//Constructor & Destructor
	BlochSimulator(){};
	~BlochSimulator(){};
	
	// Init field strength and volume
	void Initialize(ReferenceVolume* refVolume);

	// Simulation functions
	void RunSimulation(ExternalSequence& sequence, MatrixXcd& kSpace);

	//Update A matrix
	void SetRFPulse(Matrix3d& A, double rfAmplitude, double rfPhase);
	void SetOffresonance(Matrix3d& A, double dw);

	// Apply event that is the same in the entire volume e.g. relaxation phase, non-selective pulse
	void ApplyGlobalEventToVolume(SeqBlock* seqBlock);
	void ApplyEventToVolume(SeqBlock* seqBlock);
	void AcquireKSpaceLine(MatrixXcd& kSpace, SeqBlock* seqBlock, unsigned int ky);

	// appy bloch simulation to pixel
	void ApplyBlochSimulationPixel(unsigned int row, unsigned int col, Matrix3d A, double t);
};