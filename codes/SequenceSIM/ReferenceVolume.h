/* 
Class for Input Reference Volume
X by Y by 3 Volume with 2D image for proton density, R1 and R2

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
	MatrixXd R1;
	MatrixXd R2;
	
	unsigned int numberOfColumns;
	unsigned int numberOfRows;
	double       pixelSize;  // in m

public:
	ReferenceVolume()
	{
		pixelSize = 1e-3; // default
	}

	~ReferenceVolume()
	{
	};

	void AllocateMemory(unsigned int rows, unsigned int cols)
	{
		numberOfColumns = cols;
		numberOfRows = rows;
		protonDensity.resize(numberOfRows, numberOfColumns);
		protonDensity.fill(0.0);
		R1.resize(numberOfRows, numberOfColumns);
		R1.fill(0.0);
		R2.resize(numberOfRows, numberOfColumns);
		R2.fill(0.0);
	}

	void SetProtonDensityValue(unsigned int row, unsigned int col, double pdVal)
	{
		protonDensity(row, col) = pdVal;
	}

	void SetR1Value(unsigned int row, unsigned int col, double R1Val)
	{
		R1(row, col) = R1Val;
	}

	void SetR2Value(unsigned int row, unsigned int col, double R2Val)
	{
		R2(row, col) = R2Val;
	}

	void SetVolume(double** pd, double** r1, double** r2)
	{
		for (unsigned int col = 0; col < numberOfColumns; col++) {
			for (unsigned int row = 0; row < numberOfRows; col++) {
				protonDensity(row,col) = pd[row][col];
				R1(row,col) = r1[row][col];
				R2(row,col) = r2[row][col];
			}
		}
	}

	void SetPixelSize(double newPixelSize)
	{
		pixelSize = newPixelSize;
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

	double GetProtonDensityValue(unsigned int row, unsigned int col)
	{
		return protonDensity(row,col);
	}

	double GetR1Value(unsigned int row, unsigned int col)
	{
		return R1(row, col);
	}

	double GetR2Value(unsigned int row, unsigned int col)
	{
		return R2(row, col);
	}

	double GetPixelSize()
	{
		return pixelSize;
	}
};