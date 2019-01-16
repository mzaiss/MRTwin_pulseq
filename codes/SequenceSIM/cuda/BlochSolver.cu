
#include "BlochSolver.cuh"


// global host variables
bool volumeAllocated = false;
bool volumeInitialized = false;

__device__ const double two_pi = 6.283185307179586476925286766558;

__global__ void Precess(double* d_Mx, double* d_My, double* d_Mz, B_eff* d_B, Tissue* d_Tissue, double t)
{
	// get position in image from threadIdx
	int spinIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int imgPos = blockIdx.x;

	if (d_Tissue[imgPos].M0 > 0)
	{
		//get Beff Vector
		double Bx = d_B[imgPos].x*t;
		double By = d_B[imgPos].y*t;
		double Bz = d_B[imgPos].z*t;
		double b, c, s, k, nx, ny, nz;
		b = sqrt(Bx*Bx + By*By + Bz*Bz);
		if (b > 0.0)
		{

			// solve equation
			Bx /= b; nx = d_Mx[spinIdx];
			By /= b; ny = d_My[spinIdx];
			Bz /= b; nz = d_Mz[spinIdx];

			c = sin(0.5*b); c = 2.0*c*c;
			s = sin(b);
			k = nx * Bx + ny * By + nz * Bz;

			d_Mx[spinIdx] += (Bx*k - nx)*c + (ny*Bz - nz * By)*s;
			d_My[spinIdx] += (By*k - ny)*c + (nz*Bx - nx * Bz)*s;
			d_Mz[spinIdx] += (Bz*k - nz)*c + (nx*By - ny * Bx)*s;
		}
	}
}

__global__ void Dephase(double* d_Mx, double* d_My, B_eff* d_B, Tissue* d_Tissue, double t)
{
	// get position in image from threadIdx
	int spinIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int imgPos = blockIdx.x;

	if (d_Tissue[imgPos].M0 > 0)
	{
		//get Beff Vector
		double b, bz, c, s, nx, ny;
		bz = (d_B[imgPos].z + (double(threadIdx.x) / blockDim.x)*two_pi*d_Tissue[imgPos].R2) *t;
		b = fabs(bz);
		if (b > 0.0)
		{

			// solve equation
			nx = d_Mx[spinIdx];
			ny = d_My[spinIdx];
			bz /= b;

			c = sin(0.5*b); c = 2.0*c*c;
			s = sin(b);

			d_Mx[spinIdx] += (-nx)*c + ny * bz * s;
			d_My[spinIdx] += (-ny)*c + (-nx * bz)*s;
		}
	}
}

__global__ void Relax(double* d_Mx, double* d_My, double* d_Mz, Tissue* d_Tissue, double t)
{
	int spinIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int imgPos = blockIdx.x;
	
	if (d_Tissue[imgPos].M0 > 0)
	{
		double e2 = exp(-d_Tissue[imgPos].R2*t);
		double e1 = exp(-d_Tissue[imgPos].R1*t);
		d_Mx[spinIdx] *= e2;
		d_My[spinIdx] *= e2;
		d_Mz[spinIdx] = d_Tissue[imgPos].M0 + e1 * (d_Mz[spinIdx] - d_Tissue[imgPos].M0);
	}
}

__global__ void ResetBeff(B_eff* d_B)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_B[idx].x = 0;
	d_B[idx].y = 0;
	d_B[idx].z = 0;
}

__global__ void ResetInitialMagnetization(double* d_Mx, double* d_My, double* d_Mz, Tissue* d_Tissue)
{
	int spinIdx = threadIdx.x + blockIdx.x * blockDim.x;
	d_Mx[spinIdx] = 0;
	d_My[spinIdx] = 0;
	d_Mz[spinIdx] = d_Tissue[blockIdx.x].M0;
}


__global__ void ResetTransversalMagnetization(double* d_Mx, double* d_My)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_Mx[idx] = 0;
	d_My[idx] = 0;
}

__global__ void ResetKSpace(KSpaceSample* d_kSpace)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_kSpace[idx].real = 0.0;
	d_kSpace[idx].imag = 0.0;
}

__global__ void UpdateGradients(B_eff* d_B, double xGradAmplitude, double yGradAmplitude, double pixelSize)
{
	int row = threadIdx.x;
	int col = blockIdx.x;
	int idx = row + col * blockDim.x;
	d_B[idx].z += (xGradAmplitude * col + yGradAmplitude * row) * pixelSize; // remember in main program *2 * PI;
}

__global__ void UpdateRFPulse(B_eff* d_B, double rfAmplitude, double rfPhase, double offResonance)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// REMEMBER 2*PI in main Program
	d_B[idx].x = rfAmplitude * cos(rfPhase);
	d_B[idx].y = rfAmplitude * sin(rfPhase);
	d_B[idx].z += offResonance;
}

// host functions in the cu file start with cuda from now on
bool cudaInitCUDADevice()
{
	return (cudaSetDevice(0) == cudaSuccess);
}

bool cudaSyncCUDADevice()
{
	return (cudaDeviceSynchronize() == cudaSuccess);
}


bool cudaAllocateVolume(B_eff*& d_B, double*& d_Mx, double*& d_My, double*& d_Mz,
						Tissue*& d_Tissue, KSpaceSample*& d_kSpace,
	                    double*& d_MxSum, double*& d_MySum)
{
	volumeAllocated = (cudaMalloc((void**)&d_B, PIXELS * sizeof(B_eff)) == cudaSuccess &&
		cudaMalloc((void**)&d_Mx, PIXELS*SPINS * sizeof(double)) == cudaSuccess &&
		cudaMalloc((void**)&d_My, PIXELS*SPINS * sizeof(double)) == cudaSuccess &&
		cudaMalloc((void**)&d_Mz, PIXELS*SPINS * sizeof(double)) == cudaSuccess &&
		cudaMalloc((void**)&d_MxSum, PIXELS * sizeof(double)) == cudaSuccess &&
		cudaMalloc((void**)&d_MySum, PIXELS * sizeof(double)) == cudaSuccess &&
		cudaMalloc((void**)&d_Tissue, PIXELS * sizeof(Tissue)) == cudaSuccess &&
		cudaMalloc((void**)&d_kSpace, PIXELS * sizeof(KSpaceSample)) == cudaSuccess);

	return volumeAllocated;
}


bool cudaInitVolume(Tissue*& h_Tissue, Tissue*& d_Tissue, double*& d_Mx,
	double*& d_My, double*& d_Mz, B_eff*& d_B, KSpaceSample*& d_KSpace,
	double*& d_MxSum, double*& d_MySum)
{
	if (!volumeAllocated)
		return false;

	volumeInitialized = (cudaMemcpy(d_Tissue, h_Tissue, PIXELS * sizeof(Tissue), cudaMemcpyHostToDevice) == cudaSuccess);

	if (volumeInitialized)
	{
		ResetTransversalMagnetization << <ROWS, COLS >> > (d_MxSum, d_MySum);
		ResetInitialMagnetization<<<PIXELS,SPINS>>>(d_Mx, d_My, d_Mz, d_Tissue);
		ResetBeff<<<ROWS, COLS>>>(d_B);
		ResetKSpace <<<ROWS, COLS >>>(d_KSpace);
		volumeInitialized = cudaSyncCUDADevice();
	}
	return  volumeInitialized;
}


bool cudaResetBeff(B_eff*& d_B)
{
	bool isReset = false;
	if (volumeInitialized && volumeAllocated)
	{
		ResetBeff <<<ROWS, COLS >>> (d_B);
		isReset = cudaSyncCUDADevice();
	}
	return isReset;
}

bool cudaResetInitialMagnetization(double*& d_Mx, double*& d_My, double*& d_Mz, Tissue*& d_Tissue, uint32_t spins)
{
	bool isReset = false;
	if (volumeInitialized && volumeAllocated)
	{
		ResetInitialMagnetization << <PIXELS, SPINS >> > (d_Mx, d_My, d_Mz, d_Tissue);
		isReset = cudaSyncCUDADevice();
	}
	return isReset;
}

bool cudaResetKSpace(KSpaceSample*& d_kSpace)
{
	bool isReset = false;
	if (volumeInitialized && volumeAllocated)
	{
		ResetKSpace << <ROWS, COLS >> > (d_kSpace);
		isReset = cudaSyncCUDADevice();
	}
	return isReset;
}

bool cudaUpdateRFPulse(B_eff*& d_B, double rfAmplitude, double rfPhase, double Offresonance)
{
	bool isSet = false;
	if (volumeInitialized && volumeAllocated)
	{
		UpdateRFPulse <<<ROWS, COLS >>> (d_B, rfAmplitude, rfPhase, Offresonance);
		isSet = cudaSyncCUDADevice();
	}
	return isSet;
}

bool cudaUpdateGradients(B_eff*& d_B, double xGradient, double yGradient, double pixelSize)
{
	bool isSet = false;
	if (volumeInitialized && volumeAllocated)
	{
		UpdateGradients << <ROWS, COLS >> > (d_B, xGradient, yGradient, pixelSize);
		isSet = cudaSyncCUDADevice();
	}
	return isSet;
}


bool cudaSolveBlochEquation(B_eff*& d_B, double*& d_Mx, double*& d_My, double*& d_Mz, 
							Tissue*& d_Tissue, BlochSolverType type, double t)
{
	bool ranBlochEq = false;
	if (volumeInitialized && volumeAllocated)
	{
		switch (type)
		{
		case PRECESS:
			Precess<<<PIXELS, SPINS >>>(d_Mx, d_My, d_Mz, d_B, d_Tissue, t);
			ranBlochEq = cudaSyncCUDADevice();
			break;
		case RELAX:
			Relax <<<PIXELS, SPINS >>> (d_Mx, d_My, d_Mz, d_Tissue, t);
			ranBlochEq = cudaSyncCUDADevice();
			break;
		case DEPHASE:
			Dephase <<<PIXELS, SPINS >>> (d_Mx, d_My, d_B, d_Tissue, t);
			ranBlochEq = cudaSyncCUDADevice();
			break;
		default:
			break;
		}
	}
	return ranBlochEq;
}


bool cudaReturnResults(KSpaceSample*& d_kSpace, KSpaceSample*& h_kSpace)
{
	bool isSet = false;
	if (volumeInitialized && volumeAllocated)
	{
		isSet = (cudaMemcpy(h_kSpace, d_kSpace, PIXELS * sizeof(KSpaceSample), cudaMemcpyDeviceToHost) == cudaSuccess);
	}
	return isSet;
}


bool cudaReturnDoubleVector(double*& d_M, double*& h_M)
{
	bool isSet = false;
	if (volumeInitialized && volumeAllocated)
	{
		isSet = (cudaMemcpy(h_M, d_M, PIXELS * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
	}
	return isSet;
}

bool cudaReturnBeff(B_eff*& d_B, B_eff*& h_B)
{
	bool isSet = false;
	if (volumeInitialized && volumeAllocated)
	{
		isSet = (cudaMemcpy(d_B, h_B, PIXELS * sizeof(B_eff), cudaMemcpyDeviceToHost) == cudaSuccess);
	}
	return isSet;
}

bool cudaFreeVariables(B_eff*& d_B, double*& d_Mx, double*& d_My, double*& d_Mz,
						Tissue*& d_Tissue, KSpaceSample*& d_kSpace, double*& d_MxSum, double*& d_MySum)
{
	bool isFree = false;
	if (volumeInitialized && volumeAllocated)
	{
		isFree = (cudaFree(d_B) == cudaSuccess &&
			cudaFree(d_Mx) == cudaSuccess &&
			cudaFree(d_My) == cudaSuccess &&
			cudaFree(d_Mz) == cudaSuccess &&
			cudaFree(d_MxSum) == cudaSuccess &&
			cudaFree(d_MySum) == cudaSuccess &&
			cudaFree(d_Tissue) == cudaSuccess &&
			cudaFree(d_kSpace) == cudaSuccess);
	}
	return isFree;
}


bool cudaResetDevice()
{
	return (cudaDeviceReset() == cudaSuccess);
}



// k-space summation
__device__ void warpReduce(volatile double* s_data, int tid)
{
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}


// Spins are set fix to 256!!!
__global__ void SumSpins(double* d_Min, double* d_Mout)
{
	extern __shared__ double s_data[];
	unsigned int tid = threadIdx.x;
	unsigned i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	s_data[tid] = d_Min[i] + d_Min[i+blockDim.x];
	__syncthreads();

	// do sum
	if (tid < 128) {
		s_data[tid] += s_data[tid + 128];
		__syncthreads();
	}
	if (tid < 64) {
		s_data[tid] += s_data[tid + 64];
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(s_data, tid);
	}

	if (tid == 0){
		d_Mout[blockIdx.x] = s_data[0];
	}
}

__global__ void SampleKSpace(double* d_Mx, double* d_My, KSpaceSample* d_kSpace, uint32_t idx)
{
	int tid = threadIdx.x;
	if (blockIdx.x == 0)
	{
		// do sum
		if (tid < 512) {
			d_Mx[tid] += d_Mx[tid + 512];
			__syncthreads();
		}
		if (tid < 256) {
			d_Mx[tid] += d_Mx[tid + 256];
			__syncthreads();
		}
		if (tid < 128) {
			d_Mx[tid] += d_Mx[tid + 128];
			__syncthreads();
		}
		if (tid < 64) {
			d_Mx[tid] += d_Mx[tid + 64];
			__syncthreads();
		}
		if (tid < 32) {
			warpReduce(d_Mx, tid);
		}
		// return result
		if (tid == 0) {
			d_kSpace[idx].real = d_Mx[0];
		}
	}
	else if (blockIdx.x == 1)
	{
		// do sum
		if (tid < 512) {
			d_My[tid] += d_My[tid + 512];
			__syncthreads();
		}
		if (tid < 256) {
			d_My[tid] += d_My[tid + 256];
			__syncthreads();
		}
		if (tid < 128) {
			d_My[tid] += d_My[tid + 128];
			__syncthreads();
		}
		if (tid < 64) {
			d_My[tid] += d_My[tid + 64];
			__syncthreads();
		}
		if (tid < 32) {
			warpReduce(d_My, tid);
		}
		// return result
		if (tid == 0) {
			d_kSpace[idx].imag = d_My[0];
		}
	}
}

bool cudaSampleKSpace(double*& d_MxIn, double*& d_MyIn, KSpaceSample*& d_kSpace, uint32_t idx, double*& d_MxOut, double*& d_MyOut)
{
	SumSpins << <PIXELS/2, SPINS, SPINS * sizeof(double) >> > (d_MxIn, d_MxOut); // sum spins in pixel
	SumSpins << <PIXELS/2, SPINS, SPINS * sizeof(double) >> > (d_MyIn, d_MyOut); // sum spins in pixel
	SampleKSpace << <2, PIXELS >> > (d_MxOut, d_MyOut, d_kSpace, idx); // sum pixels

	return cudaSyncCUDADevice();
}

