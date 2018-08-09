/*
BlochSimulator.cpp

Class for Bloch Simulations

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#include "BlochSimulator.h"
#include "BlochEquationSolver.h"


void BlochSimulator::Initialize(ReferenceVolume* refVolume)
{
	referenceVolume = refVolume;
	Mx = MatrixXd::Zero(referenceVolume->GetNumberOfRows(), referenceVolume->GetNumberOfColumns());
	My = Mx;
	Mz = referenceVolume->GetProtonDensityMap(); // init z mag with proton density
}

void BlochSimulator::RunSimulation(ExternalSequence& sequence, MatrixXcd& kSpace)
{
	//linear reordering 
	unsigned int ky = 0;
	// loop through event blocks
	for (unsigned int nSample = 0; nSample < sequence.GetNumberOfBlocks(); nSample++)
	{
		// get current event block
		SeqBlock* seqBlock = sequence.GetBlock(nSample);
		
		// check if this is an ADC event
		if (seqBlock->isADC()) {
			AcquireKSpaceLine(kSpace, seqBlock, ky);
			ky++;
		}
        else if(~seqBlock->isADC() && (seqBlock->isTrapGradient(0)|| seqBlock->isTrapGradient(1))) {
            // Gradients needs to be calculated for each pixel
			ApplyEventToVolume(seqBlock); 
		}
		else { // No Gradients ? -> run faster global function
			ApplyGlobalEventToVolume(seqBlock);
		}
	}
}

void BlochSimulator::AcquireKSpaceLine(MatrixXcd& kSpace, SeqBlock* seqBlock, unsigned int ky)
{
	Matrix3d A = Matrix3d::Zero();
	unsigned int numCols = referenceVolume->GetNumberOfColumns();
	unsigned int numRows = referenceVolume->GetNumberOfRows();
	double pixelSize = referenceVolume->GetPixelSize();

	double GradientTimeStep = (seqBlock->GetDuration()*1e-6) / seqBlock->GetADCEvent().numSamples;
	for (unsigned int adcSample = 0; adcSample < seqBlock->GetADCEvent().numSamples; adcSample++){
		////////////////////////////////////////////////////////////////////////////////////////
		// TODO: Parallelize
		////////////////////////////////////////////////////////////////////////////////////////
		for (unsigned int col = 0; col < numCols; col++) {
			SetOffresonance(A, seqBlock->GetGradEvent(0).amplitude * col * pixelSize * TWO_PI);
			for (unsigned int row = 0; row < numRows; row++) {
				if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
					ApplyBlochSimulationPixel(row, col, A, GradientTimeStep);
				}
			}
		}
		kSpace(ky, adcSample) = std::complex<double>(Mx.sum(), My.sum());
	}
}

void BlochSimulator::ApplyGlobalEventToVolume(SeqBlock* seqBlock)
{
	Matrix3d A = Matrix3d::Zero();
	unsigned int numRows = referenceVolume->GetNumberOfRows();
	unsigned int numCols = referenceVolume->GetNumberOfColumns();

	//set the rf pulse
	SetRFPulse(A, seqBlock->GetRFEvent().amplitude*Gamma, seqBlock->GetRFEvent().phaseOffset);

	////////////////////////////////////////////////////////////////////////////////////////
	// TODO: Parallelize
	////////////////////////////////////////////////////////////////////////////////////////
	for (unsigned int row = 0; row < numRows; row++) {
		for (unsigned int col = 0; col < numCols; col++) {
			if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
				ApplyBlochSimulationPixel(row, col, A, seqBlock->GetDuration()*1e-6);
			}
		}
	}
}

void BlochSimulator::ApplyEventToVolume(SeqBlock* seqBlock)
{
	////////////////////////////////////////////////////////////////////////////////////////
	// TODO: 
	// 1. Compensate for different gradient (x,y) and pulse durations in the event duration
	// 2. Compensate for gradient rise and fall time
	////////////////////////////////////////////////////////////////////////////////////////
	Matrix3d A = Matrix3d::Zero();
	unsigned int numRows = referenceVolume->GetNumberOfRows();
	unsigned int numCols = referenceVolume->GetNumberOfColumns();
	double pixelSize = referenceVolume->GetPixelSize();
	double phaseGradientAtPx; // phase gradient at the current position

	//set the rf pulse
	SetRFPulse(A, seqBlock->GetRFEvent().amplitude*Gamma, seqBlock->GetRFEvent().phaseOffset);

	////////////////////////////////////////////////////////////////////////////////////////
	// TODO: Parallelize
	////////////////////////////////////////////////////////////////////////////////////////
	for (unsigned int row = 0; row < numRows; row++) {
		phaseGradientAtPx = seqBlock->GetGradEvent(1).amplitude * row * pixelSize * TWO_PI;
		for (unsigned int col = 0; col < numCols; col++) {
			if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
				SetOffresonance(A, seqBlock->GetGradEvent(0).amplitude * col * pixelSize * TWO_PI +phaseGradientAtPx);
				ApplyBlochSimulationPixel(row, col, A, seqBlock->GetDuration()*1e-6);
			}
		}
	}
}


void BlochSimulator::SetRFPulse(Matrix3d& A, double rfAmplitude, double rfPhase)
{
	double w1cp = rfAmplitude * cos(rfPhase);
	double w1sp = rfAmplitude * sin(rfPhase);
	A(0, 2) = -w1sp;
	A(2, 0) = w1sp;
	A(1, 2) = -w1cp;
	A(2, 1) = w1cp;
}

void BlochSimulator::SetOffresonance(Matrix3d& A, double dw)
{
	A(0, 1) = dw;
	A(1, 0) = -dw;
}

void BlochSimulator::ApplyBlochSimulationPixel(unsigned int row, unsigned int col, Matrix3d A, double t)
{
	double R1 = referenceVolume->GetR1Value(row, col);
	double R2 = referenceVolume->GetR2Value(row, col);
	A(0, 0) = -R2;
	A(1, 1) = -R2;
	A(2, 2) = -R1;
	Vector3d C(0.0, 0.0, referenceVolume->GetProtonDensityValue(row, col)*R1);
	Vector3d Mi(Mx(row, col), My(row, col), Mz(row, col));
	Vector3d M = SolveBlochEquation(Mi, A, C, t);
	Mx(row, col) = M.x();
	My(row, col) = M.y();
	Mz(row, col) = M.z();
}