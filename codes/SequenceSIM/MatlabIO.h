/*
MatlabIO.h

Class for Matlab communication

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#include "BlochSimulator.h"
#include <matrix.h>
#include <mex.h>


void ReadMATLABInput(int nrhs, const mxArray *prhs[], ReferenceVolume* refVol, BlochSimulator* blochSim, ExternalSequence* seq)
{
	if (nrhs < 2){
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:nrhs",
			"Three Inputs required: RefVolume and PulseSeq filename");
	}

	//Input 1: 3d Ref volume NxMx3
	// (:,:,1): Proton Density
	// (:,:,2): T1
	// (:,:,3): T2
	mwSize numDims = mxGetNumberOfDimensions(prhs[0]);
	if (numDims != 3)
	{
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
			"Input Volume must be 3 dimensional");
	}
	const mwSize* dims = mxGetDimensions(prhs[0]);
	if (dims[2] != 3)
	{
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
			"Input Volume must include PD, T1 and T2");
	}
	int nCols = dims[0];
	int nRows = dims[1];

	if (nCols != nRows){
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs", "Only MxM k-space possible yet (No MxN).");
	}


	//get the data for the reference volume from the matlab pointer and store it in the eigen matrix class
	refVol->AllocateMemory(nRows, nCols);
	double * pData = mxGetPr(prhs[0]);
	double t1, t2;
	for (int x = 0; x < nCols; x++){
		for (int y = 0; y < nRows; y++){
			refVol->SetProtonDensityValue(y, x, pData[x + y*nCols + 0 * (nCols*nRows)]);
			t1 = pData[x + y*nCols + 1 * (nCols*nRows)];
			t2 = pData[x + y*nCols + 2 * (nCols*nRows)];
			refVol->SetR1Value(y, x, t1 <= 0.0 ? 0.0 : 1.0 / t1);
			refVol->SetR2Value(y, x, t2 <= 0.0 ? 0.0 : 1.0 / t2);
		}
	}

	// Input 2: Filename of the pulseseq file
	const int charBufferSize = 2048;
	char tmpCharBuffer[charBufferSize];
	// gete filename from matlab
	mxGetString(prhs[1], tmpCharBuffer, charBufferSize);
	std::string seqFileName = std::string(tmpCharBuffer);
	//load the seq file
	if (!seq->load(seqFileName)) {
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
			"Seq filename not found");
	}

	// check if seq file is valid for simulation
	bool pixelSizeSet = false;
	unsigned int numberOfADCEvents = 0;
	for (unsigned int nSample = 0; nSample < seq->GetNumberOfBlocks(); nSample++)
	{
		// get current event block
		SeqBlock* seqBlock = seq->GetBlock(nSample);
		//check if it consists arbitrary gradients
		if (seqBlock->isArbitraryGradient(0) || seqBlock->isArbitraryGradient(1)){
			mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs", "Arbitrary Gardient simulation is not implemented yet");
		}
		// try to get pixel size from seq file
		if (seqBlock->isADC())
		{
			numberOfADCEvents++;
			if (seqBlock->GetADCEvent().numSamples != refVol->GetNumberOfRows())
			{
				mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
					"Mismatch between ADC samples in the sequence file and k-space samples");
			}
			if (~pixelSizeSet) {
				// Gradient amplitude (Hz/m) * Gradient Flat Time (s) = (Number Of Samples In Read Direction)/FOV (1/m)
				// -> inverse is pixel size (m)
				double pxSizeInMeter = 1.0 / (seqBlock->GetGradEvent(0).amplitude*seqBlock->GetGradEvent(0).flatTime*1e-6);
				refVol->SetPixelSize(pxSizeInMeter);
				pixelSizeSet = true;
			}
        }
	}
	if (numberOfADCEvents != refVol->GetNumberOfRows())
	{
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs", 
			"Mismatch between ADC events in the sequence file and available k-space lines");
	}
	blochSim->Initialize(refVol);
}



void ReturnKSpaceToMATLAB(int nlhs, mxArray* plhs[], MatrixXcd& kSpace)
{
	//get size for matlab pointer
	unsigned int cols = kSpace.cols();
	unsigned int rows = kSpace.rows();

	//init and set the matlab pointer
	plhs[0] = mxCreateDoubleMatrix(cols, rows, mxCOMPLEX);
	double* rOut = mxGetPr(plhs[0]);
	double* iOut = mxGetPi(plhs[0]);

	//copy kspace to matlab
	for (unsigned int y = 0; y < rows; y++){
		for (unsigned int x = 0; x < cols; x++){
			rOut[x + rows*y] = (kSpace.real())(y, x);
			iOut[x + rows*y] = (kSpace.imag())(y, x);
		}
	}
}