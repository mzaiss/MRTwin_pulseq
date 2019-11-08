
//-------------------------------------------------------------------------------------
//  Copyright (C) Fraunhofer MEVIS 2014 All Rights Reserved. Confidential
//-------------------------------------------------------------------------------------
//
//     Project: NUMARIS/4
//        File: \n4\pkg\MrServers\MrVista\Ice\IceApplicationFunctors\IceResolve\DataExportFunctor.h
//     Version:
//      Author: David Porter
//        Date: 
//        Lang: C++
//
// Description: 
//
//     Classes:
//
//-------------------------------------------------------------------------------------


//================================================================================
// Start of ResolveGateway_h wrapper
//================================================================================
#ifndef DataExportFunctor_h
#define DataExportFunctor_h 1


//================================================================================
// Includes
//================================================================================
// Import/Export DLL macro
#include "MrServers/MrVista/Ice/IceIdeaFunctors/dllInterface.h"

// base class
#include "MrServers/MrVista/include/Ice/IceUtils/IceScanFunctors.h"

// IceStorage
#include "MrServers/MrVista/include/Ice/IceBasic/IceStorage.h"

// RESOLVE parameter definitions
#include "DataExportDefs.h"


//================================================================================
// Start of project-specific namespace
//================================================================================
namespace PROJECT_NAMESPACE
{

//================================================================================
// Class: ResolveGateway
//================================================================================
class ICEIDEAFUNCTORS_API DataExportFunctor : public IceScanFunctors
{
	public:
	
        //------------------------------------------------------------------------
        // Define available properties
        //------------------------------------------------------------------------
        ICE_SCAN_FUNCTOR( DataExportFunctor )

        /*BEGIN_PROPERTY_MAP   ( ResolveGateway                        )	
            PROPERTY_DEFAULT ( AccelFactPE, "MEAS.sPat.lAccelFactPE" )
            PROPERTY_ENTRY   ( PhaseCorrWholeEchoTrain               )
            PROPERTY_ENTRY   ( ImageRecon                            )
            PROPERTY_ENTRY   ( SingleSliceRecon                      )
            PROPERTY_ENTRY   ( SliceNumber                           )

        END_PROPERTY_MAP()

        DECL_GET_SET_PROPERTY ( bool   , b, PhaseCorrWholeEchoTrain )
        DECL_GET_SET_PROPERTY ( int32_t, n, AccelFactPE             )
        DECL_GET_SET_PROPERTY ( bool   , b, ImageRecon              )
        DECL_GET_SET_PROPERTY ( bool   , b, SingleSliceRecon        )
        DECL_GET_SET_PROPERTY ( int32_t, n, SliceNumber             )*/

		//------------------------------------------------------------------------
		// Constructor / Destructor
		//------------------------------------------------------------------------
		DataExportFunctor();
		virtual ~DataExportFunctor();

		//------------------------------------------------------------------------
		// Callbacks
		//------------------------------------------------------------------------
		virtual IResult EndInit( IParcEnvironment* env );
		virtual IResult FirstCall( IceAs&, MdhProxy&, ScanControl& ctrl );
		virtual IResult endOfJob(IResult reason);

		//------------------------------------------------------------------------
		// Event sink 
		//------------------------------------------------------------------------
		virtual IResult ComputeScan ( IceAs& srcAs, MdhProxy& aMdh, ScanControl& ctrl );

	protected:

		//------------------------------------------------------------------------
		// Ice objects and IceAs
		//------------------------------------------------------------------------

		//------------------------------------------------------------------------
		// Standard variables
		//------------------------------------------------------------------------

		//------------------------------------------------------------------------
		// FILE pointers
		//------------------------------------------------------------------------

		// export data file
		FILE *m_pfExportData;

		// export parameter file
		FILE *m_pfExportParam;

		//------------------------------------------------------------------------
		// Member functions
		//------------------------------------------------------------------------

    private:
        
		//------------------------------------------------------------------------
		// Copy constructors
		//------------------------------------------------------------------------
        DataExportFunctor( const DataExportFunctor &right );
        DataExportFunctor & operator=( const DataExportFunctor &right );
};

//================================================================================
// End of project-specific namespace
//================================================================================
}

//================================================================================
// End of DataExportFunctor_h wrapper
//================================================================================
#endif