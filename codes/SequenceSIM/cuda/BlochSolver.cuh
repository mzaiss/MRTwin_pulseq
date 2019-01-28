#pragma once

//TODO: Make this object oriented
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h> // for uint32_t

const uint32_t SPINS = 256;
const uint32_t PIXELS = 1024;
const uint32_t COLS = 32;
const uint32_t ROWS = 32;

// Container for Beff vector
struct B_eff {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
};


struct Tissue {
	double R1;
	double R2;
	double M0;
};

struct KSpaceSample {
	double real = 0.0;
	double imag = 0.0;
};

enum BlochSolverType
{
	PRECESS,
	RELAX,
	DEPHASE
};



// host functions in the cu file start with cuda from now on
bool cudaInitCUDADevice();

bool cudaSyncCUDADevice();

bool cudaAllocateVolume(B_eff*& d_B, double*& d_Mx, double*& d_My, double*& d_Mz,
	Tissue*& d_Tissue, KSpaceSample*& d_kSpace,	double*& d_MxSum, double*& d_MySum);

bool cudaInitVolume(Tissue*& h_Tissue, Tissue*& d_Tissue, double*& d_Mx, double*& d_My, 
	double*& d_Mz, B_eff*& d_B, KSpaceSample*& d_KSpace, double*& d_MxSum, double*& d_MySum);

bool cudaResetBeff(B_eff*& d_B);

bool cudaResetInitialMagnetization(double*& d_Mx, double*& d_My, double*& d_Mz, Tissue*& d_Tissue, uint32_t spins);

bool cudaResetKSpace(KSpaceSample*& d_kSpace);

bool cudaUpdateRFPulse(B_eff*& d_B, double rfAmplitude, double rfPhase, double Offresonance);

bool cudaUpdateGradients(B_eff*& d_B, double xGradient, double yGradient, double pixelSize);

bool cudaSolveBlochEquation(B_eff*& d_B, double*& d_Mx, double*& d_My, double*& d_Mz,
	Tissue*& d_Tissue, BlochSolverType type, double t);

bool cudaReturnResults(KSpaceSample*& d_kSpace, KSpaceSample*& h_kSpace);

bool cudaFreeVariables(B_eff*& d_B, double*& d_Mx, double*& d_My, double*& d_Mz,
	Tissue*& d_Tissue, KSpaceSample*& d_kSpace, double*& d_MxSum, double*& d_MySum);

bool cudaResetDevice();

bool cudaReturnDoubleVector(double*& d_M, double*& h_M);

bool cudaReturnBeff(B_eff*& d_B, B_eff*& h_B);

bool cudaSampleKSpace(double*& d_MxIn, double*& d_MyIn, KSpaceSample*& d_kSpace, uint32_t idx, double*& d_MxOut, double*& d_MyOut);
