/*
MatlabIO.h

Class for Matlab communication

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#include "ReferenceVolume.h"
#include "BlochSimulator.h"
#include <matrix.h>
#include <mex.h>

void ReadMATLABInput(int nrhs, const mxArray *prhs[], ReferenceVolume* refVol, BlochSimulator* blochSim)
{
	if (nrhs < 3){
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:nrhs",
			"Three Inputs required, RefVolume, PulseTrain, B1");
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

	//get the data for the reference volume from the matlab pointer and store it in the eigen matrix class
	refVol->AllocateMemory(nCols, nRows);
	double * pData = mxGetPr(prhs[0]);
	for (int x = 0; x < nCols; x++){
		for (int y = 0; y < nRows; y++){
			refVol->SetProtonDensityValue(y, x, pData[x + y*nCols + 0 * (nCols*nRows)]);
			refVol->SetT1Value(y, x, pData[x + y*nCols + 1 * (nCols*nRows)]);
			refVol->SetT2Value(y, x, pData[x + y*nCols + 2 * (nCols*nRows)]);
		}
	}

	//Input 2: 2D pulse vector (Mx6)
	// (:,1): Pulse magnitude
	// (:,2): Pulse phase
	// (:,3): X gradient
	// (:,4); Y Gradient
	// (:,5): Timesteps
	// (:,6): ADC
	numDims = mxGetNumberOfDimensions(prhs[1]);
	if (numDims != 2)
	{
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
			"Input Pulse Train must be 2 dimensional");
	}
	const mwSize* dimsVec = mxGetDimensions(prhs[1]);
	if (dimsVec[1] != 6)
	{
		mexErrMsgIdAndTxt("MRIzero:ReadMATLABInput:rrhs",
			"Input Pulse Train must include mag, phase, xgrad, ygrad, t and adc");
	}

	//Input 3: Field Strength
	// double B0 [T]
	double B0 = mxGetScalar(prhs[2]);
	unsigned int numPulseSamples = dimsVec[0];

	//init the simulator
	blochSim->Initialize(B0, refVol);
	blochSim->AllocateMemory(numPulseSamples);
    
	//set the pulse, gradient and adc events
	double * pPulseData = mxGetPr(prhs[1]);
	for (unsigned int i = 0; i < numPulseSamples; i++)
	{
		blochSim->SetRFPulses(i, pPulseData[i + numPulseSamples * 0], pPulseData[i + numPulseSamples * 1]);
		blochSim->SetGradients(i, pPulseData[i + numPulseSamples * 2], pPulseData[i + numPulseSamples * 3]);
		blochSim->SetTimesteps(i, pPulseData[i + numPulseSamples * 4]);
		blochSim->SetADC(i, (bool)pPulseData[i + numPulseSamples * 5]);
	}
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
	for (unsigned int x = 0; x < cols; x++){
		for (unsigned int y = 0; y < rows; y++){
			rOut[y + cols*x] = (kSpace.real())(y, x);
			iOut[y + cols*x] = (kSpace.imag())(y, x);
		}
	}
}