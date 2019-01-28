/* 
Class for Input Reference Volume
X by Y by 3 Volume with 2D image for proton density, R1 and R2

MRIzero Project

kai.herz@tuebingen.mpg.de
*/
#include <cstring> // for memcpy

#pragma once

class ReferenceVolume
{
private:
	

    double* M0;
	double* R1;
	double* R2;
	
	unsigned int numberOfColumns;
	unsigned int numberOfRows;
	unsigned int numberOfPixels;
	const double pixelSize = 0.22/32;  // FOV is fixed to 220mm and resolution is fixed to 32x32

	bool refVolAllocated = false;

public:
	ReferenceVolume()
	{
	}

	~ReferenceVolume()
	{
		if (refVolAllocated)
		{
			delete M0; M0 = nullptr;
			delete R1; R1 = nullptr;
			delete R2; R2 = nullptr;

		}
	};

	unsigned int ImageToArrayPosition(unsigned int row, unsigned int col) { return row + col * numberOfRows; }

	void AllocateMemory(unsigned int rows, unsigned int cols)
	{
		numberOfColumns = cols;
		numberOfRows = rows;
		numberOfPixels = numberOfColumns * numberOfRows;
		M0 = new double[numberOfPixels];
		R1 = new double[numberOfPixels];
		R2 = new double[numberOfPixels];
		refVolAllocated = true;
	}

	void SetProtonDensityValue(unsigned int row, unsigned int col, double pdVal)
	{
		M0[ImageToArrayPosition(row, col)] = pdVal;
	}

	void SetR1Value(unsigned int row, unsigned int col, double R1Val)
	{
		R1[ImageToArrayPosition(row, col)] = R1Val;
	}

	void SetR2Value(unsigned int row, unsigned int col, double R2Val)
	{
		R2[ImageToArrayPosition(row, col)] = R2Val;
	}

	void SetVolume(double* pd, double* r1, double* r2)
	{
		memcpy(M0, pd, numberOfPixels * sizeof(double));
		memcpy(R1, r1, numberOfPixels * sizeof(double));
		memcpy(R1, r2, numberOfPixels * sizeof(double));
	}

	unsigned int GetNumberOfRows()
	{
		return numberOfRows;
	}

	unsigned int GetNumberOfColumns()
	{
		return numberOfColumns;
	}

	double* GetProtonDensityMap()
	{
		return M0;
	}

	double* GetR1Map()
	{
		return R1;
	}

	double* GetR2Map()
	{
		return R2;
	}


	double GetProtonDensityValue(unsigned int row, unsigned int col)
	{
		return M0[ImageToArrayPosition(row, col)];
	}

	double GetProtonDensityValue(unsigned int idx)
	{
		return M0[idx];
	}

	double GetR1Value(unsigned int row, unsigned int col)
	{
		return R1[ImageToArrayPosition(row, col)];
	}

	double GetR1Value(unsigned int idx)
	{
		return R1[idx];
	}

	double GetR2Value(unsigned int row, unsigned int col)
	{
		return R2[ImageToArrayPosition(row, col)];
	}

	double GetR2Value(unsigned int idx)
	{
		return R2[idx];
	}

	double GetPixelSize()
	{
		return pixelSize;
	}
};