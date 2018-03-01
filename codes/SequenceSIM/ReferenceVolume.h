/* 
Class for Input Reference Volume
X by Y by 3 Volume with 2D image for proton density, T1 and T2

MRIzero Project

kai.herz@tuebingen.mpg.de
*/

#pragma once

#include "Eigen\eigen"

using namespace Eigen;

class ReferenceVolume
{
private:
	
    MatrixXd protonDensity;
	MatrixXd T1;
	MatrixXd T2;
	
	unsigned int numberOfColumns;
	unsigned int numberOfRows;
	double       pixelSize;  // in mm
	bool isAllocated;

public:
	ReferenceVolume()
	{
		pixelSize = 1; // for now
		isAllocated = false;
	}

	~ReferenceVolume()
	{
	};

	void AllocateMemory(unsigned int x, unsigned int y)
	{
		numberOfColumns = x;
		numberOfRows = y;
		protonDensity.resize(numberOfRows, numberOfColumns);
		protonDensity.fill(0.0);
		T1.resize(numberOfRows, numberOfColumns);
		T1.fill(0.0);
		T2.resize(numberOfRows, numberOfColumns);
		T2.fill(0.0);
		isAllocated = true;
	}

	void SetProtonDensityValue(unsigned int row, unsigned int col, double pdVal)
	{
		protonDensity(row, col) = pdVal;
	}

	void SetT1Value(unsigned int row, unsigned int col, double t1Val)
	{
		T1(row, col) = t1Val;
	}

	void SetT2Value(unsigned int row, unsigned int col, double t2Val)
	{
		T2(row, col) = t2Val;
	}

	void SetVolume(double** pd, double** t1, double** t2)
	{
		for (unsigned int col = 0; col < numberOfColumns; col++) {
			for (unsigned int row = 0; row < numberOfRows; col++) {
				protonDensity(row,col) = pd[row][col];
				T1(row,col) = t1[row][col];
				T2(row,col) = t2[row][col];
			}
		}
	}

	unsigned int GetNumberOfRows()
	{
		return numberOfRows;
	}

	unsigned int GetNumberOfColumns()
	{
		return numberOfColumns;
	}

	MatrixXd GetProtonDensityMap()
	{
		return protonDensity;
	}

	double GetProtonDensityValue(unsigned int x, unsigned int y)
	{
		return protonDensity(y,x);
	}

	double GetT1Value(unsigned int x, unsigned int y)
	{
		return T1(y,x);
	}

	double GetT2Value(unsigned int x, unsigned int y)
	{
        return T2(y,x);
	}

	double GetPixelSize()
	{
		return pixelSize;
	}
};