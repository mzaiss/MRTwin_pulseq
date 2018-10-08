/*
BlochSimulator.cpp

Class for Bloch Simulations

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#include "BlochSimulator.h"
#include "BlochEquationSolver.h"


void BlochSimulator::Initialize(ReferenceVolume* refVolume, unsigned int numSamples)
{
	referenceVolume = refVolume;
	Mx = MatrixXd::Zero(referenceVolume->GetNumberOfRows(), referenceVolume->GetNumberOfColumns());
	My = Mx;
	Mz = referenceVolume->GetProtonDensityMap(); // init z mag with proton density
    kx = 0;
    ky = 0;
	numKSpaceSamples = numSamples;
}

void BlochSimulator::RunSimulation(ExternalSequence& sequence, std::vector<KSpaceEvent>& kSpace)
{
	unsigned int currentADC = 0;
	// loop through event blocks
	for (unsigned int nSample = 0; nSample < sequence.GetNumberOfBlocks(); nSample++)
	{
		// get current event block
		SeqBlock* seqBlock = sequence.GetBlock(nSample);
		
		// check if this is an ADC event
		if (seqBlock->isADC()) {
			AcquireKSpaceLine(kSpace, seqBlock, currentADC);
		}
        
        // Pseudo-Spoiler (NOT PHYSICALLY REASONABLE!)
        // set Mx=My=0 if any z gradient is applied
        else if(seqBlock->isTrapGradient(2)) {
            //Mx = MatrixXd::Zero(referenceVolume->GetNumberOfRows(), referenceVolume->GetNumberOfColumns());
            //My = Mx;
            for (unsigned int row = 0; row < referenceVolume->GetNumberOfRows(); row++) {
                for (unsigned int col = 0; col < referenceVolume->GetNumberOfColumns(); col++) {
                    Mx(row, col) = 0;
                    My(row, col) = 0;
                }
            }
        }
        else if(~seqBlock->isADC() && (seqBlock->isTrapGradient(0)|| seqBlock->isTrapGradient(1))) {
            // Gradients needs to be calculated for each pixel
			ApplyEventToVolume(seqBlock); 
			// Todo: take ramp times into account
            ky += (seqBlock->GetGradEvent(1).flatTime) * (seqBlock->GetGradEvent(1).amplitude)*1e-6;
            kx += (seqBlock->GetGradEvent(0).flatTime) * (seqBlock->GetGradEvent(0).amplitude)*1e-6;
		}
		else { // No Gradients ? -> run faster global function
			ApplyGlobalEventToVolume(seqBlock);
			if(seqBlock->isRF()) {
				// check if 180 pulse
	            if (fabs((seqBlock->GetRFEvent().amplitude)*(seqBlock->GetDuration()*1e-6) - 0.5) < 0.01) { 
	                kx = -kx;
	                ky = -ky;
	            }
	            else {
	                kx = 0;
	                ky = 0;
	            }
			}
		}
		delete seqBlock; // pointer gets allocate with new in the GetBlock() function
	}
}

void BlochSimulator::AcquireKSpaceLine(std::vector<KSpaceEvent>& kSpace, SeqBlock* seqBlock, unsigned int &currentADC)
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
		double xGrad = seqBlock->GetGradEvent(0).amplitude * col * pixelSize * TWO_PI;
			for (unsigned int row = 0; row < numRows; row++) {
				SetOffresonance(A, seqBlock->GetGradEvent(1).amplitude * row * pixelSize * TWO_PI + xGrad);
				if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
					ApplyBlochSimulationPixel(row, col, A, GradientTimeStep, DEPHASE);
				}
			}
		}
		// update gradients
		kx += GradientTimeStep * seqBlock->GetGradEvent(0).amplitude;
		ky += GradientTimeStep * seqBlock->GetGradEvent(1).amplitude;
		//sample data
		kSpace[currentADC].kSample = std::complex<double>(Mx.sum(), My.sum());
		kSpace[currentADC].kX = kx;
		kSpace[currentADC].kY = ky;
		currentADC++;
	}
}

void BlochSimulator::ApplyGlobalEventToVolume(SeqBlock* seqBlock)
{
	Matrix3d A = Matrix3d::Zero();
	unsigned int numRows = referenceVolume->GetNumberOfRows();
	unsigned int numCols = referenceVolume->GetNumberOfColumns();

	
	////////////////////////////////////////////////////////////////////////////////////////
	// TODO: Parallelize
	////////////////////////////////////////////////////////////////////////////////////////
	if (seqBlock->isRF())
	{
		//set the rf pulse
	    SetRFPulse(A, seqBlock->GetRFEvent().amplitude*TWO_PI, seqBlock->GetRFEvent().phaseOffset, seqBlock->GetRFEvent().freqOffset*TWO_PI);
		for (unsigned int row = 0; row < numRows; row++) {
			for (unsigned int col = 0; col < numCols; col++) {
				if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
					ApplyBlochSimulationPixel(row, col, A, seqBlock->GetDuration()*1e-6, PRECESS);
				}
			}
		}
	}
	else
	{
		for (unsigned int row = 0; row < numRows; row++) {
			for (unsigned int col = 0; col < numCols; col++) {
				if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
					ApplyBlochSimulationPixel(row, col, A, seqBlock->GetDuration()*1e-6, RELAX);
				}
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
	SetRFPulse(A, seqBlock->GetRFEvent().amplitude*TWO_PI, seqBlock->GetRFEvent().phaseOffset, seqBlock->GetRFEvent().freqOffset*TWO_PI);

	////////////////////////////////////////////////////////////////////////////////////////
	// TODO: Parallelize
	////////////////////////////////////////////////////////////////////////////////////////
	for (unsigned int row = 0; row < numRows; row++) {
		phaseGradientAtPx = seqBlock->GetGradEvent(1).amplitude * row * pixelSize * TWO_PI;
		for (unsigned int col = 0; col < numCols; col++) {
			if (referenceVolume->GetProtonDensityValue(row, col) > 0) { // skip if there is no tissue
				SetOffresonance(A, seqBlock->GetGradEvent(0).amplitude * col * pixelSize * TWO_PI +phaseGradientAtPx);
				ApplyBlochSimulationPixel(row, col, A, seqBlock->GetDuration()*1e-6, PRECESS);
			}
		}
	}
}


void BlochSimulator::SetRFPulse(Matrix3d& A, double rfAmplitude, double rfPhase, double rfFreqOffset)
{
	double w1cp = rfAmplitude * cos(rfPhase);
	double w1sp = rfAmplitude * sin(rfPhase);
	A(0, 2) = -w1sp;
	A(2, 0) = w1sp;
	A(1, 2) = -w1cp;
	A(2, 1) = w1cp;
    
    SetOffresonance(A, rfFreqOffset);
}

void BlochSimulator::SetOffresonance(Matrix3d& A, double dw)
{
	A(0, 1) = dw;
	A(1, 0) = -dw;
}

void BlochSimulator::ApplyBlochSimulationPixel(unsigned int row, unsigned int col, Matrix3d& A, double t, BlochSolverType type)
{
	double R1 = referenceVolume->GetR1Value(row, col);
	double R2 = referenceVolume->GetR2Value(row, col);
	A(0, 0) = -R2;
	A(1, 1) = -R2;
	A(2, 2) = -R1;
	Vector3d C(0.0, 0.0, referenceVolume->GetProtonDensityValue(row, col)*R1);
	Vector3d Mi(Mx(row, col), My(row, col), Mz(row, col));
	Vector3d M;
	switch (type)
	{
	case FULL:
		M = SolveBlochEquation(Mi, A, C, t);
		break;
	case PRECESS:
		M = Precess(Mi, A, t);
		break;
	case RELAX:
		M = Relax(Mi, A, referenceVolume->GetProtonDensityValue(row, col), t);
		break;
    case DEPHASE:
		M = Dephase(Mi, A, t);
		break;
	default:
		break;
	}
	Mx(row, col) = M.x();
	My(row, col) = M.y();
	Mz(row, col) = M.z();
}