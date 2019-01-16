/* 
BlochSimulator.h

Class for Bloch Simulations

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#pragma once

#include "ReferenceVolume.h"
#include "ExternalSequence.h"
#include "BlochSolver.cuh"

class KSpaceEvents
{
public:
	KSpaceSample* kSample;
	double* kX;
	double* kY;
	uint32_t numberOfSamples;

	KSpaceEvents(uint32_t nSamples)
	{
		numberOfSamples = nSamples;
		kSample = new KSpaceSample[numberOfSamples];
		kX = new double[numberOfSamples];
		kY = new double[numberOfSamples];
	}

	~KSpaceEvents()
	{
		delete kSample;
		delete kX;
		delete kY;
	}
};



class BlochSimulator
{
private:

    ReferenceVolume* referenceVolume; // ref volume with t1,t2 and pd  
	double kx, ky;					 
	uint32_t numKSpaceSamples;
	bool lastCudaError;
	double pixelSize;

	/////////// DEVICE VARIABLES
	double* d_Mx; // all spins (rows*cols*spins)
	double* d_My;
	double* d_Mz;
	double* d_MxSum; // sum of all spins (rows*cols)
	double* d_MySum;

	B_eff*  d_Beff;
	Tissue* d_Tissue;
	KSpaceSample* d_kSpace;


public:
	//Constructor & Destructor
	BlochSimulator(){};
	~BlochSimulator(){};
	
	// Init field strength and volume
    bool Initialize(ReferenceVolume* refVolume)
	{
		referenceVolume = refVolume;
		pixelSize = referenceVolume->GetPixelSize();
		kx = 0;
		ky = 0;
		bool cudaInit = cudaInitCUDADevice();
		if (cudaInit) {
			cudaInit = cudaAllocateVolume(d_Beff, d_Mx, d_My, d_Mz, d_Tissue, d_kSpace, d_MxSum, d_MySum);
		}

		if (cudaInit) {
			Tissue* h_Tissue = new Tissue[PIXELS];
			for (uint32_t i = 0; i < PIXELS; i++)
			{
				h_Tissue[i].M0 = referenceVolume->GetProtonDensityValue(i);
				h_Tissue[i].R1 = referenceVolume->GetR1Value(i);
				h_Tissue[i].R2 = referenceVolume->GetR2Value(i);

			}
			cudaInit = cudaInitVolume(h_Tissue, d_Tissue, d_Mx, d_My, d_Mz, 
				d_Beff, d_kSpace, d_MxSum, d_MySum);
			delete h_Tissue;

		}
		return cudaInit;
	}


	// Simulation functions
	void RunSimulation(ExternalSequence& sequence, KSpaceEvents& kSpace)
	{
		uint32_t currentADC = 0;
		// loop through event blocks
		for (uint32_t nSample = 0; nSample < sequence.GetNumberOfBlocks(); nSample++)
		{
			// get current event block
			SeqBlock* seqBlock = sequence.GetBlock(nSample);

			//TODO: implement checks
			lastCudaError = cudaResetBeff(d_Beff); // reset magnetization

			// check if this is an ADC event
			if (seqBlock->isADC()) {
				AcquireKSpaceLine(kSpace, seqBlock, currentADC);
			}
			else if (~seqBlock->isADC() && (seqBlock->isTrapGradient(0) || seqBlock->isTrapGradient(1))) {
				// Gradients needs to be calculated for each pixel
				ApplyEventToVolume(seqBlock);
				// Todo: take ramp times into account
				ky += (seqBlock->GetGradEvent(1).flatTime) * (seqBlock->GetGradEvent(1).amplitude)*1e-6* referenceVolume->GetPixelSize();
				kx += (seqBlock->GetGradEvent(0).flatTime) * (seqBlock->GetGradEvent(0).amplitude)*1e-6* referenceVolume->GetPixelSize();
			}
			else { // No Gradients ? -> run faster global function
				ApplyGlobalEventToVolume(seqBlock);
				if (seqBlock->isRF()) {
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

		lastCudaError = cudaReturnResults(d_kSpace, kSpace.kSample);
		lastCudaError = cudaSyncCUDADevice();
		lastCudaError = cudaFreeVariables(d_Beff, d_Mx, d_My, d_Mz, d_Tissue, d_kSpace, d_MxSum, d_MySum);
		lastCudaError = cudaResetDevice();
	}

	// Apply event that is the same in the entire volume e.g. relaxation phase, non-selective pulse
	void ApplyGlobalEventToVolume(SeqBlock* seqBlock)
	{
		lastCudaError = cudaUpdateRFPulse(d_Beff, seqBlock->GetRFEvent().amplitude*TWO_PI, seqBlock->GetRFEvent().phaseOffset, seqBlock->GetRFEvent().freqOffset*TWO_PI);
		lastCudaError = cudaSolveBlochEquation(d_Beff, d_Mx, d_My, d_Mz, d_Tissue, seqBlock->isRF() ? PRECESS : RELAX, seqBlock->GetDuration()*1e-6);
	}

	void ApplyEventToVolume(SeqBlock* seqBlock)
	{
		////////////////////////////////////////////////////////////////////////////////////////
		// TODO: 
		// 1. Compensate for different gradient (x,y) and pulse durations in the event duration
		// 2. Compensate for gradient rise and fall time
		////////////////////////////////////////////////////////////////////////////////////////

		//set the gradients
		lastCudaError = cudaUpdateGradients(d_Beff, seqBlock->GetGradEvent(0).amplitude * TWO_PI, seqBlock->GetGradEvent(1).amplitude * TWO_PI, referenceVolume->GetPixelSize());
		lastCudaError = cudaSolveBlochEquation(d_Beff, d_Mx, d_My, d_Mz, d_Tissue, DEPHASE, seqBlock->GetDuration()*1e-6);
	}

	void AcquireKSpaceLine(KSpaceEvents& kSpace, SeqBlock* seqBlock, uint32_t &currentADC)
	{
		double t = (seqBlock->GetDuration()*1e-6) / seqBlock->GetADCEvent().numSamples;
		double xGrad2PI = seqBlock->GetGradEvent(0).amplitude* TWO_PI;
		double yGrad2PI = seqBlock->GetGradEvent(1).amplitude* TWO_PI;
		double xGradStep = seqBlock->GetGradEvent(0).amplitude * t * referenceVolume->GetPixelSize();
		double yGradStep = seqBlock->GetGradEvent(1).amplitude * t * referenceVolume->GetPixelSize();
		lastCudaError = cudaResetBeff(d_Beff);
		lastCudaError = cudaUpdateGradients(d_Beff, xGrad2PI, yGrad2PI, pixelSize);
		for (uint32_t adcSample = 0; adcSample < seqBlock->GetADCEvent().numSamples; adcSample++) {
			lastCudaError = cudaSolveBlochEquation(d_Beff, d_Mx, d_My, d_Mz, d_Tissue, DEPHASE, t);
			// update gradients
			kx += xGradStep;
			ky += yGradStep;
			//sample data
			lastCudaError = cudaSampleKSpace(d_Mx, d_My, d_kSpace, currentADC, d_MxSum, d_MySum);
			kSpace.kX[currentADC] = kx;
			kSpace.kY[currentADC] = ky;
			currentADC++;
		}
	}
};
