
//-------------------------------------------------------------------------------------
//  Copyright (C) Fraunhofer MEVIS 2014 All Rights Reserved. Confidential
//-------------------------------------------------------------------------------------
//
//     Project: NUMARIS/4
//        File: \n4\pkg\MrServers\MrVista\Ice\IceApplicationFunctors\IceResolve\DataExportFunctor.cpp
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
// Defines to activate traces
//================================================================================
#define TRACE_ComputeScan

//================================================================================
// Defines that modify functor behaviour
//================================================================================


//================================================================================
// Includes
//================================================================================
#include "DataExportFunctor.h"
#include <stdio.h> 

//================================================================================
// Start of project-specific namespace
//================================================================================
namespace PROJECT_NAMESPACE
{

//================================================================================
// Function:        DataExportFunctor()
//
// Description:     Default Constructor
//================================================================================
DataExportFunctor::DataExportFunctor()	
{ 
	//----------------------------------------------------------------------------
	// Initialise member variables
	//----------------------------------------------------------------------------

	//----------------------------------------------------------------------------
	// Register the callback function
	//----------------------------------------------------------------------------
    addCB( IFirstCall );
}


//================================================================================
// Function:        ~DataExportFunctor()
//
// Description:     Default Destructor
//================================================================================
DataExportFunctor::~DataExportFunctor()
{}


//================================================================================
// Function:        EndInit()
//
// Description:     Default Destructor
//================================================================================
IResult DataExportFunctor::EndInit( IParcEnvironment* /*env*/ )
{
	ICE_SET_FN("DataExportFunctor::EndInit( IParcEnvironment* env )")

	return I_OK;

} // EndInit()


//================================================================================
// Function:        FirstCall()
//
// Description:     Callback function
//================================================================================
IResult DataExportFunctor::FirstCall( IceAs& /*srcAs*/, MdhProxy& /*aMdh*/, ScanControl& /*ctrl*/ )
{
	//----------------------------------------------------------------------------
	// Local variables
	//----------------------------------------------------------------------------
	ICE_SET_FN("DataExportFunctor::FirstCall()")

	//----------------------------------------------------------------------------
	// Open file for data export
	//----------------------------------------------------------------------------
	char FileExportData[128];

	sprintf( FileExportData, "%s.xpt", FILENAME_EXPORT );

	m_pfExportData = fopen( FileExportData, "w" );

	//----------------------------------------------------------------------------
	// Open file for data export
	//----------------------------------------------------------------------------
	char FileExportParam[128];

	sprintf( FileExportParam, "%s.par", FILENAME_EXPORT );

	m_pfExportParam = fopen( FileExportParam, "w" );

	//----------------------------------------------------------------------------
	// Finish
	//----------------------------------------------------------------------------
	return I_OK;

} // FirstCall()


//================================================================================
// Function:        ComputeScan()
//
// Description:     Event sink
//================================================================================
IResult DataExportFunctor::ComputeScan ( IceAs& srcAs, MdhProxy& aMdh, ScanControl& ctrl )
{
	//----------------------------------------------------------------------------
	// Local variables
	//----------------------------------------------------------------------------
	ICE_SET_FN("DataExportFunctor::ComputeScan(srcAs, dataHeader, ctrl)")

	IResult res = I_FAIL;

	CMPLX *P1 = NULL;

	//----------------------------------------------------------------------------
	// Execute callbacks - e.g. FirstCall()
	//----------------------------------------------------------------------------
    res = ExecuteCallbacks( srcAs, aMdh, ctrl );

    if( failed( res ) )
    {
        ICE_RET_ERR("ExecuteCallbacks failed, aborting...", res)
    }

    //------------------------------------------------------------------
    // Pass phase corection scans straight to next functor
    //------------------------------------------------------------------
	if ( aMdh.getEvalInfoMask() & MDH_PHASCOR )
	{
		// trace

		#ifdef TRACE_ComputeScan

			ICE_OUT ( "Passing phase-correction scan straight to next functor" );

		#endif

		// call successor

		/*res = ScanReady( srcAs, aMdh, ctrl );

		if ( failed( res ) )
		{
			ICE_RET_ERR("ScanReady() failed for phase correction scan, aborting...", res)
		}*/

		// finish

		return I_OK;
	}

	//----------------------------------------------------------------------------
	// Pass iPAT ref scans straight to next functor
	//----------------------------------------------------------------------------
	if ( aMdh.isPatRefScan() )
	{	
		// trace

        //fprintf(m_pfExportParam, "\n %d \t %d \n", aMdh.getClin(), aMdh.getCpar());

        //P1 = (CMPLX*)srcAs.calcSplObjStartAddr();

        //for (long lCha = 0; lCha < ctrl.m_imaDimBoundaries.m_ccha; lCha++)
        //{
        //    /*fprintf( m_pfExportParam, "CHANNEL: %d\n\n", lCha );*/

        //    for (long lCol = 0; lCol < ctrl.m_imaDimBoundaries.m_ccol; lCol++)
        //    {
        //        fprintf(m_pfExportData, "%le\t%le\n", P1->re, P1->im);

        //        /*fprintf( m_pfExportParam, "COLUMN:%d\t", lCol );
        //        fprintf( m_pfExportParam, "%le\t%le\n", P1->re, P1->im );*/

        //        ++P1;
        //    }
        //}
        /*aMdh.addToEvalInfoMask(MDH_SKIP_ONLINE_PHASCOR);
        aMdh.addToEvalInfoMask(MDH_SKIP_REGRIDDING);*/
		/*#ifdef TRACE_ComputeScan

			ICE_OUT ( "Passing iPAT ref scan straight to next functor" );

		#endif*/

		// call successor

		/*res = ScanReady( srcAs, aMdh, ctrl );

		if ( failed( res ) )
		{
			ICE_RET_ERR("ScanReady() failed for PAT REF scan, aborting...", res)
		}*/

		// finish

		return I_OK;
    }

	//----------------------------------------------------------------------------
	// Export data
	//----------------------------------------------------------------------------
	/*if ( aMdh.getCeco() == 0 )
	{*/
        fprintf(m_pfExportParam, "\n %d \t %d \n", aMdh.getClin(), aMdh.getCpar());
        
		P1 = (CMPLX*) srcAs.calcSplObjStartAddr(); 

		for ( long lCha = 0 ; lCha < ctrl.m_imaDimBoundaries.m_ccha ; lCha++ )
		{
			/*fprintf( m_pfExportParam, "CHANNEL: %d\n\n", lCha );*/

			for ( long lCol = 0 ; lCol < ctrl.m_imaDimBoundaries.m_ccol ; lCol++ )
			{
				fprintf( m_pfExportData, "%le\t%le\n", P1->re, P1->im );

				/*fprintf( m_pfExportParam, "COLUMN:%d\t", lCol );
				fprintf( m_pfExportParam, "%le\t%le\n", P1->re, P1->im );*/

				++P1;
			}
		}
	//}
ICE_OUT(FILENAME_EXPORT);
	//----------------------------------------------------------------------------
	// Call next functor
	//----------------------------------------------------------------------------

    // trace

    #ifdef TRACE_ComputeScan

	   // ICE_OUT_PARAM ( "Passing imaging scans to next functor" );

    #endif

    // pass imaging scans to next functor in chain

	/*res = ScanReady( srcAs, aMdh, ctrl );

	if ( failed( res ) )
	{
		ICE_RET_ERR("ScanReady() failed, aborting...", res)
	}*/

	//----------------------------------------------------------------------------
	// Finish
	//----------------------------------------------------------------------------
	return I_OK;

} // ComputeScan()


//================================================================================
// Function:        endOfJob()
//
// Description:     Standard 'Callback' function
//================================================================================
IResult DataExportFunctor::endOfJob(IResult reason)
{
    ICE_SET_FN("DataExportFunctor::endOfJob()")

	//----------------------------------------------------------------------------
	// Check if acquisition has ended normally
	//---------------------------------------------------------------------------- 
	if( reason != I_ACQ_END )   
	{
        // trace
		ICE_OUT("Acquisition has ended abnormally" );
	}

	else
	{
		ICE_OUT("Acquisition has ended normally" );
	}

	//---------------------------------------------------------------------------- 
	// Clean up PARC environment
	//---------------------------------------------------------------------------- 

	//---------------------------------------------------------------------------- 
	// Close files
	//---------------------------------------------------------------------------- 
	fclose( m_pfExportData );
	fclose( m_pfExportParam );

	//---------------------------------------------------------------------------- 
	// Finish
	//----------------------------------------------------------------------------  
	return I_OK;

} // endOfJob()


//================================================================================
// End of project-specific namespace
//================================================================================
}