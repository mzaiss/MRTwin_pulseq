
//-------------------------------------------------------------------------------------
//  Copyright (C) Fraunhofer MEVIS 2014 All Rights Reserved. Confidential
//-------------------------------------------------------------------------------------
//
//     Project: NUMARIS/4
//        File: \n4\pkg\MrServers\MrVista\Ice\IceApplicationFunctors\IceResolve\DataExportConfigurator.cpp
//     Version:
//      Author: David Porter
//        Date: 
//        Lang: C++
//
// Description: 
//
//     Classes: DataExportConfigurator
//
//-------------------------------------------------------------------------------------


//================================================================================
// Includes
//================================================================================

// DataExportConfigurator
#include "DataExportConfigurator.h"

// //ICE_OUT, ICE_ERR
#include "MrServers/MrVista/include/Parc/Trace/IceTrace.h"

// IceExtractMode, PaFTModes
#include "MrServers/MrVista/include/Ice/IceDefs.h"

// Protbasic
#include "ProtBasic/Interfaces/FilterDefines.h"
#include "ProtBasic/Interfaces/KSpaceDefines.h"
#include "ProtBasic/Interfaces/MdsDefines.h"
#include "ProtBasic/Interfaces/NavigatorDefines.h"
#include "ProtBasic/Interfaces/PatDefines.h"
#include "ProtBasic/Interfaces/PhysioDefines.h"
#include "ProtBasic/Interfaces/ProtDataDefines.h"
#include "ProtBasic/Interfaces/SeqExpoDefines.h"
#include "ProtBasic/Interfaces/TXSpecDefines.h"
#include "ProtBasic/Interfaces/VectorDefines.h"
#include "ProtBasic/Interfaces/ApplDefines.h"
#include "ProtBasic/Interfaces/MeasAqcDefs.h"

// ICE_RET_ERR
#include "MrServers/MrVista/include/IrisICE.h"

// vector
#include <vector>


//================================================================================
// Namespaces
//================================================================================
using namespace MED_MRES_XProtocol;  //namespace of everything exported by XProtocol component
using namespace std;


//================================================================================
// Start of SEQ_NAMESPACE
//================================================================================
namespace PROJECT_NAMESPACE
{

//================================================================================
// Compose()
//================================================================================
IResult DataExportConfigurator::Compose( ProtocolComposer::Toolbox& toolbox )
{
	//----------------------------------------------------------------------------
	// Local variables and settings
	//----------------------------------------------------------------------------
	ICE_SET_FN("DataExportConfigurator::Compose");

	using namespace ProtocolComposer;
	toolbox.Init( "IRIS" );

	//----------------------------------------------------------------------------
	// Insert and configure ResolveGateway functor
	//----------------------------------------------------------------------------

	// find Flags functor as anchor

	MrFunctor funAnchor = toolbox.FindFunctor( ANCHOR_FUNCTOR );

	// find name of corresponding pipe service

	std::string sPipeAnchor = funAnchor.PipeServiceName();

    ICE_OUT("PipeAnchor: " << sPipeAnchor);
	// clone pipe service

	MrPipeService pipeAnchor = toolbox.FindPipeService( sPipeAnchor.c_str() );
 
	// create DataExportFunctor functor
    ICE_OUT("PipeAnchor Find");
	char tDataExportFunctor[64];
	sprintf( tDataExportFunctor, "DataExportFunctor@%s", ICE_DLL_NAME );

	MrFunctor funDataExport = pipeAnchor.CreateFunctor("DataExportFunctor", tDataExportFunctor );
    ICE_OUT("Export Create");
	if ( !funDataExport.IsOk() )
	{
		ICE_RET_ERR("Failed to create funDataExport, aborting...\n", I_FAIL);
	}

	// insert DataExportFunctor into preconfigured path

	toolbox.InsertAfter("DataExportFunctor","ComputeScan","ScanReady",ANCHOR_FUNCTOR,"ScanReady");
    ICE_OUT("Export Insert");
	//----------------------------------------------------------------------------
	// ORIGINAL CODE
	//----------------------------------------------------------------------------

    /*int32_t nSetMeas                    = -1;
    int32_t nRepMeas                    = -1;
    int32_t nSegMeas                    = -1;
    int32_t nChaMeas                    = -1;
    int32_t nAveMeas                    = -1;
    int32_t nRawCols                    = -1;
    int32_t nRawColsOS2                 = -1;
    int32_t nAddCols                    = -1;
    int32_t nAddColsOS2                 = -1;
    int32_t nRawLines                   = -1;
    int32_t nNavRawLines                = -1;
    int32_t nContrasts                  = -1;
    int32_t nShots                      = -1;
    int32_t nSlices                     = -1;
    int32_t nEchoSpacing                = -1;
    int32_t nTR                         = -1;
    int32_t nDiffusionMode              = -1;
    int32_t nDiffScans                  = -1;
    int32_t nReAcquisitionMode          = -1;

	long lROSegments     = -1;
	long lBaseResolution = -1;
    
	unsigned ucROPartialFourierFactor = 0;

	bool bInterpolation2D = false;

	std::string sPipeName;

	//Make a local copy of the XProtocol
	XProtocol& prot = toolbox.Protocol();

	//----------------------------------------------------------------------------
	// Check some assumptions
	//----------------------------------------------------------------------------
	if (  prot["YAPS.flReadoutOSFactor"].ToLong() != 4 )
	{
		ICE_RET_ERR("Readout oversampling factor not set to 4, aborting...\n", I_FAIL);
	}

    //----------------------------------------------------------------------------
    // Initialise some parameters
    //----------------------------------------------------------------------------

    // readout segments

    lROSegments = prot["MEAS.sFastImaging.lReadoutSegments"].ToLong();

    // readout partial Fourier
    
    ucROPartialFourierFactor = prot["MEAS.sKSpace.ucReadoutPartialFourier"].ToLong();

    if ( ucROPartialFourierFactor != SEQ::PF_OFF )
    {   
        ICE_OUT( "RESOLVE INFO: measurement performed using readout partial Fourier\n" );
    }

    // shots

    nShots = calcNumberOfShots( lROSegments, ucROPartialFourierFactor );

    // number of additional columns (no oversampling)

    nAddCols = RESOLVE_ADDITIONAL_COLUMNS;

    // number of additional colums (2x oversampling)

    nAddColsOS2 = nAddCols * 2;

	// base resolution

	lBaseResolution = prot["MEAS.sKSpace.lBaseResolution"].ToLong();

    // number of columns acquired (4x oversampling)

    nRawCols = (int32_t) ( ( ( lBaseResolution / lROSegments ) + nAddCols ) * prot["YAPS.flReadoutOSFactor"].ToLong() );

    // number of columns (2x oversampling)

    nRawColsOS2 = nRawCols / 2;

    // number of raw lines in imaging echo

    nRawLines = prot["MEAS.sFastImaging.lEPIFactor"].ToLong();

    // number of raw lines in navigator echo

    nNavRawLines = prot["MEAS.sKSpace.lLinesPerShot"].ToLong();

    // number of contrasts (spin echoes)

    nContrasts = prot["MEAS.lContrasts"].ToLong();

    // number of global averages (expected value is 1 due to b-value specific averaging in VD13A)

    nAveMeas = prot["MEAS.lAverages"].ToLong();

    if ( nAveMeas != 1 )
    {
        ICE_RET_ERR("Number of global averages not set to 1, aborting...\n", I_FAIL);
    }

    // number of repetitions (expected value is 1 due to b-value specific averaging in VD13A)

    nRepMeas = prot["MEAS.lRepetitions"].ToLong() + 1;

    if ( nRepMeas != 1 )
    {
        ICE_RET_ERR("Number of repetitions not set to 1, aborting...\n", I_FAIL);
    }

    // size of SET dimension (expected value is 1 in VD13A)

    nSetMeas = prot["YAPS.iNSet"].ToLong();

    if ( nSetMeas != 1 )
    {
        ICE_RET_ERR("Number of SET's not set to 1, aborting...\n", I_FAIL);
    }

    // number of RF receive channels

    nChaMeas = prot["YAPS.iMaxNoOfRxChannels"].ToLong();

    // number of slices

    nSlices = prot["MEAS.sSliceArray.lSize"].ToLong();

    // echo spacing

    nEchoSpacing = prot["MEAS.sFastImaging.lEchoSpacing"].ToLong();

    // repetition time

    nTR = prot["MEAS.alTR"].ToLong(0);

    // diffusion mode

    nDiffusionMode = prot["MEAS.sDiffusion.ulMode"].ToLong();

    // sinc interpolation

    bInterpolation2D = prot["MEAS.sKSpace.uc2DInterpolation"].ToBool();

    // reacquisition mode (note misspelling of this parameter name in XProtocol)

    nReAcquisitionMode = prot["MEAS.ulReaquisitionMode"].ToLong();

    // total number of diffusion scans (equal to the number of repetition indices written to YAPS by sequence)

    nDiffScans = prot[ "YAPS.alICEProgramPara" ].ToLong( ICE_PROGRAM_DIFF_REPETITIONS ); 

    if( nDiffScans <= 0 ) 
    {
        ICE_RET_ERR( "Invalid number of diffusion scans, aborting...", I_FAIL );
    }

    //----------------------------------------------------------------------------
    // Find number of available threads
    //----------------------------------------------------------------------------
    int iThreadsPerThreadPool(1);
    MrFunctor funWorkerThreadDispenser = toolbox.FindService("WorkerThreadDispenser");

    if ( funWorkerThreadDispenser.IsBad() )
    {
        ICE_ERR("Service 'WorkerThreadDispenser' not found, aborting...");
        return I_FAIL;
    }

    iThreadsPerThreadPool = funWorkerThreadDispenser.GetProperty("NumberOfProcessors").ToLong();

	//----------------------------------------------------------------------------
	// Set number of phase correction scans for Nyquist ghost phase correction.
    //
    // Note that sequence always acquires full echo-train phase correction data.
	//----------------------------------------------------------------------------
    #ifdef PHASE_CORR_WHOLE_ECHO_TRAIN

	    nSegMeas = prot["MEAS.sKSpace.lPhaseEncodingLines"].ToLong();

    #else

        nSegMeas = 2;

    #endif

	//----------------------------------------------------------------------------
	// Insert and configure ResolveFeedback functor
	//----------------------------------------------------------------------------

	// find FeedbackRoot functor as anchor

	MrFunctor funFeedbackRoot = toolbox.FindFunctor("FeedbackRoot");

	// find name of corresponding pipe service

	sPipeName = funFeedbackRoot.PipeServiceName();

	// clone pipe service

	MrPipeService pipeFeedbackRoot = toolbox.FindPipeService(sPipeName.c_str());
 
	// create ResolveFeedback functor

	char tResolveFeedback[64];
	sprintf( tResolveFeedback, "ResolveFeedback@%s", ICE_DLL_NAME );

	MrFunctor funResolveFeedback = pipeFeedbackRoot.CreateFunctor("ResolveFeedback", tResolveFeedback );

	if ( !funResolveFeedback.IsOk() )
	{
	ICE_RET_ERR("Failed to create funResolveFeedback, aborting...\n", I_FAIL);
	}

	// insert ResolveFeedback functor into preconfigured path

	funFeedbackRoot.ConnectOutput("ScanReady", "ResolveFeedback", "ComputeScan");

	// switch feedback on/off in ResolveFeedback functor

	#ifdef OFFLINE_MODE

	funResolveFeedback.SetProperty("FeedbackMode", (int32_t) FEEDBACK_OFF );

	#else

	funResolveFeedback.SetProperty("FeedbackMode", (int32_t) FEEDBACK_ON );

	#endif

	// switch reacquisition mode on/off in ResolveFeedback functor

	if ( nShots == 1 )
	{
		funResolveFeedback.SetProperty("ReAcquisitionModeOn", false );
	}

	else
	{
		if ( nReAcquisitionMode == SEQ::REACQU_MODE_ON )
		{
			funResolveFeedback.SetProperty("ReAcquisitionModeOn", true );
		}

		else
		{
			funResolveFeedback.SetProperty("ReAcquisitionModeOn", false );
		}
	}

	// set other properties of ResolveFeedback functor

	funResolveFeedback.SetProperty("Channels", nChaMeas );
	funResolveFeedback.SetProperty("RawColsOS2", nRawColsOS2 );
	funResolveFeedback.SetProperty("RawLines", nRawLines );
	funResolveFeedback.SetProperty("NavCols", nRawCols );
	funResolveFeedback.SetProperty("NavRawLines", nNavRawLines );
	funResolveFeedback.SetProperty("Contrasts", nContrasts );
	funResolveFeedback.SetProperty("ROSegments", (int32_t) lROSegments );
	funResolveFeedback.SetProperty("Shots", nShots );
	funResolveFeedback.SetProperty("Slices", nSlices );
	funResolveFeedback.SetProperty("EchoSpacing", nEchoSpacing );
	funResolveFeedback.SetProperty("TR", nTR );
	funResolveFeedback.SetProperty("Interpolation2D", bInterpolation2D );
	funResolveFeedback.SetProperty("RoSpreadThreshold", (double) RESOLVE_RO_SPREAD_THRESHOLD );
	funResolveFeedback.SetProperty("MaxTimeFactorForReAcquisition", (double) RESOLVE_MAX_TIME_FACTOR_FOR_REACQ );
	funResolveFeedback.SetProperty("MaxTimeForDynFreqScansUS", (int32_t) RESOLVE_MAX_TIME_FOR_DYN_FREQ_SCANS_US );
	funResolveFeedback.SetProperty("SkipReAcquiredData", false );
	funResolveFeedback.SetProperty("MaxRawDataSize_MB", (int32_t) RESOLVE_MAX_RAW_SIZE_ICE_SLIDING_WINDOW_MB );
	funResolveFeedback.SetProperty("Averages", nRepMeas );
	funResolveFeedback.SetProperty("DiffScans", nDiffScans );

    //----------------------------------------------------------------------------
    // Reconfigure root functor
    //----------------------------------------------------------------------------

    // find root functor

    MrFunctor funRootFunctor = toolbox.FindFunctor("RootFunctor");

    // get the name of the corresponding pipeservice

    sPipeName = funRootFunctor.PipeServiceName();

    // Clone the pipeservice

    MrPipeService pipeRootFunctor = toolbox.FindPipeService(sPipeName.c_str());

    // set number of raw columns based on 2x oversampling

    funRootFunctor.SetProperty("NImageCols", nRawColsOS2);

    // set size of SEG dimension (used by phase correction scans)

    funRootFunctor.SetProperty("NSegMeas", nSegMeas);

    // PHS dimension is used to individual shots

    funRootFunctor.SetProperty("NPhsMeas", nShots);

    // set NRepMeas to the total number of diffusion scans

    funRootFunctor.SetProperty("NRepMeas", nDiffScans );

	//----------------------------------------------------------------------------
	// Insert and configure ResolveGateway functor
	//----------------------------------------------------------------------------

	// find Flags functor as anchor

	MrFunctor funFlags = toolbox.FindFunctor("Flags");

	// find name of corresponding pipe service

	sPipeName = funFlags.PipeServiceName();

	// clone pipe service

	MrPipeService pipeFlags = toolbox.FindPipeService(sPipeName.c_str());
 
	// create ResolveGateway functor

	char tResolveGateway[64];
	sprintf( tResolveGateway, "ResolveGateway@%s", ICE_DLL_NAME );

	MrFunctor funResolveGateway = pipeFlags.CreateFunctor("ResolveGateway", tResolveGateway );

	if ( !funResolveGateway.IsOk() )
	{
		ICE_RET_ERR("Failed to create funResolveGateway, aborting...\n", I_FAIL);
	}

	// insert ResolveGateway functor into preconfigured path

	toolbox.InsertBefore("ResolveGateway", "ComputeScan", "ScanReady", "Flags", "ComputeScan");

	// set type of Nyquist ghost phase correction
	// used to select which phase correction scans get passed on to next functor and which are ignored

	#ifdef PHASE_CORR_WHOLE_ECHO_TRAIN

		funResolveGateway.SetProperty("PhaseCorrWholeEchoTrain", true );

	#else

		funResolveGateway.SetProperty("PhaseCorrWholeEchoTrain", false );

	#endif

	//----------------------------------------------------------------------------
	// Reconfigure Flags functor
	//----------------------------------------------------------------------------
	funFlags.SetProperty("NBoundPhs", nShots );

	//----------------------------------------------------------------------------
	// Reconfigure RoFT functor
	//----------------------------------------------------------------------------

	// find RoFT functor

	MrFunctor funRoFt = toolbox.FindFunctor("roft");

	// modify no. of columns in RoFT Functor to the value corresponding to OS factor 2

	funRoFt.SetProperty("NImageCols", nRawColsOS2);

	//----------------------------------------------------------------------------
	// Replace adjroftregrid functor with AdjRoFtResolve and configure
	//----------------------------------------------------------------------------

	// find adjroftregrid functor

	MrFunctor funAdjRoFt = toolbox.FindFunctor("adjroftregrid");
	
	// get the name of the corresponding pipeservice

	sPipeName = funAdjRoFt.PipeServiceName();

	//clone the pipeservice

	MrPipeService pipeAdjRoFt = toolbox.FindPipeService(sPipeName.c_str());
    
    // create ResolveAdjustRoFt functor

    char tResolveAdjRoFt[64];
    sprintf( tResolveAdjRoFt, "ResolveAdjustRoFt@%s", ICE_DLL_NAME );

    MrFunctor funResolveAdjRoFt = pipeAdjRoFt.CreateFunctor("ResolveAdjRoFt", tResolveAdjRoFt );

    if ( !funResolveAdjRoFt.IsOk() )
    {
        ICE_RET_ERR( "Failed to create funResolveAdjRoFt, aborting ... ", I_FAIL );
    }

	// set no. of columns in ResolveAdjustRoFt functor to the acquired value (with OS factor 4)

	funResolveAdjRoFt.SetProperty("NRoFtlen", nRawCols);

	// replace adjroftregrid functor with ResolveAdjustRoFt functor

	toolbox.Replace("adjroftregrid", "ResolveAdjRoFt"); 

    //----------------------------------------------------------------------------
    // Insert AdjustRoFtInputData functor
    //----------------------------------------------------------------------------

    // create a new instance of RoFT functor

    MrFunctor funAdjustRoFtInputData = pipeAdjRoFt.CreateFunctor("AdjustRoFtInputData", "AdjustRoFtInputData@IceScanFunctors");

    if ( !funAdjustRoFtInputData.IsOk() )
    {
        ICE_RET_ERR( "Failed to create funAdjustRoFtInputData, aborting ... ", I_FAIL );
    }

    // insert ResolveRoFt functor before ResolveAdjRoFt functor

    toolbox.InsertBefore("AdjustRoFtInputData", "ComputeScan", "ScanReady", "ResolveAdjRoFt", "ComputeScan");

    // set no. of columns in ResolveAdjustRoFt functor to the acquired value (with OS factor 4)

    funAdjustRoFtInputData.SetProperty("NRoFtlen",nRawCols);

    //----------------------------------------------------------------------------
    // Insert additional RoFT functor (normal FT)
    //----------------------------------------------------------------------------

    // create a new instance of RoFT functor

    MrFunctor funResolveRoFtNormal = pipeAdjRoFt.CreateFunctor("ResolveRoFtNormal", "RoFt@IceScanFunctors");

    if ( !funResolveRoFtNormal.IsOk() )
    {
        ICE_RET_ERR( "Failed to create ResolveRoFtNormal, aborting ... ", I_FAIL );
    }

    // insert ResolveRoFt functor before ResolveAdjRoFt functor

    toolbox.InsertBefore("ResolveRoFtNormal", "ComputeScan", "ScanReady", "ResolveAdjRoFt", "ComputeScan");

    // modify no. of columns in ResolveRoFT Functor to the value corresponding to OS factor 4 (original sample rate) WITH ZERO FILLING

    funResolveRoFtNormal.SetProperty("NImageCols", nRawCols);

    //----------------------------------------------------------------------------
    // Relocate onlintse functor
    //----------------------------------------------------------------------------

    // find original onlinetse functor

    MrFunctor funOnlineTse = toolbox.FindFunctor( "onlinetse", false );

    if ( !funOnlineTse.IsOk() )
    {
        ICE_RET_ERR( "Failed to find onlinetse, aborting ... ", I_FAIL );
    }

    // create a new onlinetse functor in the same pipesevice as the AdjRoFtResolve functor

    char tResolveOnlineTse[64];
    sprintf( tResolveOnlineTse, "ResolveOnlineTse@%s", ICE_DLL_NAME );

    MrFunctor funResolveOnlineTse = pipeAdjRoFt.CreateFunctor("ResolveOnlineTse", tResolveOnlineTse );

    if ( !funResolveOnlineTse.IsOk() )
    {
        ICE_RET_ERR( "Failed to create funResolveOnlineTse, aborting ... ", I_FAIL );
    }

    // insert new funResolveOnlineTse functor before ResolveAdjRoFt functor

    toolbox.InsertBefore("ResolveOnlineTse", "ComputeScan", "ScanReady", "ResolveAdjRoFt", "ComputeScan");

    // copy properties from old onlinetse functor to new funResolveOnlineTse functor

    MrProperties OnlineTseFunctorProps = funOnlineTse.Properties();
    funResolveOnlineTse.SetProperties( OnlineTseFunctorProps );

    // remove original onlinetse functor

    if ( failed( toolbox.Remove( "onlinetse" , false) ) )
    {
        ICE_RET_ERR("Failed to remove onlinetse from functor chain, aborting...\n", I_FAIL);
    }

    //----------------------------------------------------------------------------
    // Create single-threaded pipe service for ResolveOnlineTse functor
    //----------------------------------------------------------------------------

    // create single-threaded pipe service for ResolveOnlineTse functor

    MrPipeService pipeResolveOnlineTse = toolbox.SplitAfter("ResolveRoFtNormal","ResolveOnlineTse");
    pipeResolveOnlineTse.PoolThreads( 1 );

    // create single-threaded pipe service for ResolveRoFtInverse, ResolveAdjRoFt and roft functors

    MrPipeService pipeResolveAdjRoFt = toolbox.SplitAfter("ResolveOnlineTse","ResolveAdjRoFt");
    pipeResolveAdjRoFt.PoolThreads( 1 );

    //----------------------------------------------------------------------------
    // Relocate accuPhaseCorr functor
    //----------------------------------------------------------------------------

    // find original accuPhaseCorr functor

    MrFunctor funAccuPhaseCorr = toolbox.FindFunctor( "accuPhaseCorr", false );

    if ( !funAccuPhaseCorr.IsOk() )
    {
        ICE_RET_ERR( "Failed to find accuPhaseCorr, aborting ... ", I_FAIL );
    }

    // create a new accuPhaseCorr functor in the same pipesevice as the AdjRoFtResolve functor

    MrFunctor funResolveAccuPhaseCorr = pipeResolveOnlineTse.CreateFunctor("ResolveAccuPhaseCorr", "AccuDecorator@IceScanFunctors");

    if ( !funResolveAccuPhaseCorr.IsOk() )
    {
        ICE_RET_ERR( "Failed to create funResolveOnlineTse, aborting ... ", I_FAIL );
    }

    // connect new ResolveAccuPhaseCorr to ResolveOnlineTse functor

    funResolveOnlineTse.ConnectOutput( "PCReady", "ResolveAccuPhaseCorr", "ComputeScan" );

    // copy properties from old accuPhaseCorr functor to new ResolveAccuPhaseCorr functor

    MrProperties AccuPhaseCorrFunctorProps = funAccuPhaseCorr.Properties();
    funResolveAccuPhaseCorr.SetProperties( AccuPhaseCorrFunctorProps );

    // remove original accuPhaseCorr functor

    if ( failed( toolbox.Remove("accuPhaseCorr") ) )
    {
        ICE_RET_ERR("Failed to remove accuPhaseCorr from functor chain, aborting...\n", I_FAIL);
    }

    //----------------------------------------------------------------------------
    // Insert additional RoFT functor (inverse FT)
    //----------------------------------------------------------------------------

    // create a new instance of RoFT functor

    MrFunctor funResolveRoFtInverse = pipeResolveAdjRoFt.CreateFunctor("ResolveRoFtInverse", "RoFt@IceScanFunctors");

    if ( !funResolveRoFtInverse.IsOk() )
    {
    ICE_RET_ERR( "Failed to create funResolveRoFtInverse, aborting ... ", I_FAIL );
    }

    // insert ResolveRoFt functor before ResolveAdjRoFt functor

    toolbox.InsertBefore("ResolveRoFtInverse", "ComputeScan", "ScanReady", "ResolveAdjRoFt", "ComputeScan");

    // modify no. of columns in ResolveRoFT Functor to the value corresponding to OS factor 4 (original sample rate) WITH ZERO FILLING

    funResolveRoFtInverse.SetProperty("NImageCols", nRawCols);

    // specify inverse FFT

    funResolveRoFtInverse.SetProperty("FTMode", (int) FT_Inverse);

    //----------------------------------------------------------------------------
    // Remove ROFilterFunctor
    //----------------------------------------------------------------------------

	// find ROFilterFunctor

	MrFunctor funROFilterFunctor = toolbox.FindFunctor("ROFilterFunctor",false);

	// remove ROFilterFunctor functor if it is in functor chain

	if( funROFilterFunctor.IsOk() )
	{
		// connect ResolveAdjRoFt functor to roft functor (required to avoid connection error when ROFilterFunctor is deleted below)

		funResolveAdjRoFt.ConnectOutput( "ScanReady", "roft", "ComputeScan" );

		// delete ROFilterFunctor

		if ( failed( toolbox.Remove("ROFilterFunctor", false) ) )
		{
			ICE_RET_ERR("Failed to remove ROFilterFunctor from functor chain, aborting...\n", I_FAIL);
		}
	}

	//----------------------------------------------------------------------------
	// Reconfigure RawObjProvider
	//----------------------------------------------------------------------------

	// find RawObjProvider

	MrFunctor funRaw = toolbox.FindFunctor("rawobjprovider");

	// modify no. of columns in RawObjProvider to the value corresponding to OS factor 2

	funRaw.SetProperty("RawCol", nRawColsOS2);

	// PHS dimension is used to store individual shots

	funRaw.SetProperty("RawPhs", nShots);

	// set RawRep to the total number of diffusion scans

	funRaw.SetProperty("RawRep", nDiffScans);

	//----------------------------------------------------------------------------
	// Replace AccuDecorator with ResolveAccuDecorator and configure 
	//----------------------------------------------------------------------------

	// find AccuDecorator

	MrFunctor funAccuDecorator = toolbox.FindFunctor("accu");

	// get the name of the corresponding pipeservice

	sPipeName = funAccuDecorator.PipeServiceName();

	// clone the pipeservice

	MrPipeService pipeAccuDecorator = toolbox.FindPipeService(sPipeName.c_str());

	// create ResolveAccuDecorator functor

	char tResolveAccuDecorator[64];
	sprintf( tResolveAccuDecorator, "ResolveAccuDecorator@%s", ICE_DLL_NAME );

	MrFunctor funResolveAccuDecorator = pipeAccuDecorator.CreateFunctor("ResolveAccu", tResolveAccuDecorator );

	if ( !funResolveAccuDecorator.IsOk() )
	{
		ICE_RET_ERR( "Failed to create funResolveAccuDecorator, aborting ... ", I_FAIL );
	}
	
	// copy properties from AccuDecorator to ResolveAccuDecorator
	// this must be done before calling 'Replace' since 'Replace' destroys 'AccuDecorator' and all its Properties!

	MrProperties AccuDecoratorProps = funAccuDecorator.Properties();
	funResolveAccuDecorator.SetProperties( AccuDecoratorProps );    

	// replace AccuDecorator with ResolveAccuDecorator

	toolbox.Replace("accu", "ResolveAccu");

	// modify accu functor to avoid raw data averaging

	if( nAveMeas > 1 )
	{            
		funResolveAccuDecorator.SetProperty ("InplaceAccu",false);
		funResolveAccuDecorator.SetProperty ("IsDimensionAveraging",false);
		funResolveAccuDecorator.SetProperty ("ForwardChunkOfAverages",false);
	}

	//----------------------------------------------------------------------------
	// Reconfigure Check4Calculation
	//----------------------------------------------------------------------------

	// find Check4Calculation

	MrFunctor funCheck4Calculation = toolbox.FindFunctor("Check4Calculation");

	// switch off ima counter, which sets MDH_PHASEFFT flag
	// this flag should only be set by sequence to allow 'sliding-window' re-acquisition scheme

	funCheck4Calculation.SetProperty("IsImaCounterOn", false );
	funCheck4Calculation.SetProperty("IsIPatCounterOn", false );

	//----------------------------------------------------------------------------
	// Replace ImageLooper functor with ResolveDiffusionLooper and configure
	//----------------------------------------------------------------------------

	// find ImageLooper functor

	MrFunctor funImageLooper = toolbox.FindFunctor("ImageLooper");

	// get the name of the corresponding pipeservice

	sPipeName = funImageLooper.PipeServiceName();

	// clone the pipeservice

	MrPipeService pipeImageLooper = toolbox.FindPipeService(sPipeName.c_str());
    
	// create ResolveDiffusionLooper functor

	char tResolveDiffusionLooper[64];
	sprintf( tResolveDiffusionLooper, "ResolveDiffusionLooper@%s", ICE_DLL_NAME );

	MrFunctor funResolveDiffusionLooper = pipeImageLooper.CreateFunctor("ResolveDiffusionLooper", tResolveDiffusionLooper );

	if ( !funResolveDiffusionLooper.IsOk() )
	{
		ICE_RET_ERR( "Failed to create funResolveDiffusionLooper, aborting ... ", I_FAIL );
	}
	
	// copy properties from ImageLooper to ResolveDiffusionLooper
	// this must be done before calling 'Replace' since 'Replace' destroys 'ImageLooper' and all it's Properties!

	MrProperties loopersProps = funImageLooper.Properties();
	funResolveDiffusionLooper.SetProperties(loopersProps);    

	// replace ImageLooper with ResolveDiffusionLooper

	toolbox.Replace("ImageLooper", "ResolveDiffusionLooper");

	// configure looper to respond to MDH trigger events from sequence

	funResolveDiffusionLooper.SetProperty("TriggerMode", (int32_t) ICE::TriggerPerPhaseFT);

	// set property for number of raw navigator lines
    
	funResolveDiffusionLooper.SetProperty("NavRawLines", nNavRawLines );

	//----------------------------------------------------------------------------
	// Insert and configure ResolveObjProvider functor.
	//
	// Note that number of raw columns sent to ResolveObjProvider functor corresponds to
	// Readout OS factor 2 (measurement uses OS factor 4). The factor 2 allows
	// offcentre FOV to be applied within ResolveObjProvider functor after spliced image 
	// has been generated.
	//----------------------------------------------------------------------------

	// modify number of lines to take account of iPAT acceleration factor

	int32_t nRawLinesAfterGrappa = ( (nRawLines - 1) * prot["MEAS.sPat.lAccelFactPE"].ToLong() ) + 1;

	// create ResolveObjProvider functor

	char tResolveObjProvider[64];
	sprintf( tResolveObjProvider, "ResolveObjProvider@%s", ICE_DLL_NAME );

	MrFunctor funResolveObjProvider = pipeImageLooper.CreateFunctor("ResolveObjProvider", tResolveObjProvider );

	if ( !funResolveObjProvider.IsOk() )
	{
		ICE_RET_ERR( "Failed to create funResolveObjProvider, aborting ... ", I_FAIL );
	}

	// insert ResolveObjProvider into preconfigured path

	toolbox.InsertAfter("ResolveObjProvider","ComputeImage","ImageReady","ResolveDiffusionLooper","ImageReady");

	// set properties

	funResolveObjProvider.SetProperty("RawCols", nRawColsOS2 );
	funResolveObjProvider.SetProperty("RawLines", nRawLinesAfterGrappa );
	funResolveObjProvider.SetProperty("Shots", nShots );

	//----------------------------------------------------------------------------
	// Relocate GRAPPA functor
	//----------------------------------------------------------------------------

	// find original Grappa functor

	MrFunctor funGrappaFunctor = toolbox.FindFunctor( "GrappaFunctor", false );

	// if functor was found, relocate it by creating a copy and deleting original

	if ( funGrappaFunctor.IsOk() )
	{
		// create a new Grappa functor in the same pipesevice as the ResolveLooper and ResolveObjProvider functors

		MrFunctor funResolveGrappa = pipeImageLooper.CreateFunctor( "ResolveGrappa", "GrappaFunctor@IceImageReconFunctors" );

		if ( !funResolveGrappa.IsOk() )
		{
			ICE_RET_ERR( "Failed to create funResolveGrappa, aborting ... ", I_FAIL );
		}

		// insert the new Grappa functor after ResolveDiffusionLooper

		toolbox.InsertAfter( "ResolveGrappa", "ComputeImage", "ImageReady", "ResolveDiffusionLooper", "ImageReady" );

		// create new multi-threaded pipe service for new Grappa functor

		MrPipeService pipeResolveGrappa = toolbox.SplitAfter( "ResolveDiffusionLooper", "ResolveGrappa" );
		MrPipeService pipeResolveObjProvider = toolbox.SplitAfter( "ResolveGrappa", "ResolveObjProvider" );
		pipeResolveGrappa.PoolThreads( iThreadsPerThreadPool );

		// copy properties from old Grappa functor to new Grappa functor

		MrProperties GrappaFunctorProps = funGrappaFunctor.Properties();
		funResolveGrappa.SetProperties( GrappaFunctorProps );

		// remove original Grappa functor

		if ( failed( toolbox.Remove( "GrappaFunctor" ) ) )
		{
			ICE_RET_ERR( "Failed to remove GrappaFunctor from functor chain, aborting...\n", I_FAIL );
		}

		// find iPATCalcStoreIndex functor

		MrFunctor funIPATCalcStoreIndex = toolbox.FindFunctor( "iPATCalcStoreIndex" );

		if ( !funIPATCalcStoreIndex.IsOk() )
		{
			ICE_RET_ERR( "********** iPATCalcStoreIndex not found in functor chain **********\n", I_FAIL );
		}

		// get the name of the corresponding pipeservice

		sPipeName = funIPATCalcStoreIndex.PipeServiceName();

		// clone the pipeservice

		MrPipeService pipeIPATCalcStoreIndex = toolbox.FindPipeService(sPipeName.c_str());

		// create a new ResolveIPATCalcStoreIndex functor in the same pipesevice as the standard iPATCalcStoreIndex functor

		char tResolveIPATCalcStoreIndex[64];
		sprintf( tResolveIPATCalcStoreIndex, "ResolveIPATCalcStoreIndex@%s", ICE_DLL_NAME );

		MrFunctor funResolveIPATCalcStoreIndex = pipeIPATCalcStoreIndex.CreateFunctor( "ResolveIPAT", tResolveIPATCalcStoreIndex );

		if ( !funResolveIPATCalcStoreIndex.IsOk() )
		{
			ICE_RET_ERR( "Failed to create funResolveIPATCalcStoreIndex, aborting...\n", I_FAIL );
		}

		// copy properties from iPATCalcStoreIndex to ResolveIPATCalcStoreIndex

		MrProperties iPATFunctorProps = funIPATCalcStoreIndex.Properties();
		funResolveIPATCalcStoreIndex.SetProperties( iPATFunctorProps );

		// Patch connections between ResolveIPATCalcStoreIndex and other functors

		funRoFt.ConnectOutput( "ScanReady", "ResolveIPAT", "ComputeScan" );
		funResolveIPATCalcStoreIndex.ConnectOutput( "ScanReady", "rawobjprovider", "ComputeScan" );
		funResolveIPATCalcStoreIndex.ConnectOutput( "RefScanReady", "accuRefScan", "ComputeScan" );

		// remove original iPATCalcStoreIndex functor

		if ( failed( toolbox.Remove( "iPATCalcStoreIndex", false ) ) )
		{
			ICE_RET_ERR( "Failed to remove iPATCalcStoreIndex from functor chain, aborting...\n", I_FAIL );
		}

		// Debug: display PAT ref data

		#ifdef CTRL_DISPLAY_PAT_REF

			// find Check4Calculation functor

			MrFunctor funCheck4Calculation = toolbox.FindFunctor( "Check4Calculation", false );

			// error if Check4Calculation is not found

			if ( !funCheck4Calculation.IsOk() )
			{
				ICE_RET_ERR( "Check4Calculation functor not found in functor chain, aborting...\n", I_FAIL );
			}

			// get the name of the corresponding pipeservice

			sPipeName = funCheck4Calculation.PipeServiceName();

			// clone the pipeservice

			MrPipeService pipeCheck4Calculation = toolbox.FindPipeService(sPipeName.c_str());

			// create ResolveDisplayPatRef functor

			char tResolveDisplayPatRef[64];
			sprintf( tResolveDisplayPatRef, "ResolveDisplayPatRef@%s", ICE_DLL_NAME );

			MrFunctor funResolveDisplayPatRef = pipeCheck4Calculation.CreateFunctor( "ResolveDisplayPatRef", tResolveDisplayPatRef );

			if ( !funResolveDisplayPatRef.IsOk() )
			{
				ICE_RET_ERR( "Failed to create funResolveDisplayPatRef, aborting...\n", I_FAIL );
			}

			// insert ResolveDisplayPatRef functor after Check4Calculation

			toolbox.InsertAfter( "ResolveDisplayPatRef", "ComputeScan", "ScanReady", "Check4Calculation", "ScanReady" );

			// find GrappaFindWs functor

			MrFunctor funGrappaFindWs = toolbox.FindFunctor( "GrappaFindWs", false );

			// copy some properties from GrappaFindWs ResolveDisplayPatRef

			funResolveDisplayPatRef.SetProperty( "UseAllReps4Ws", funGrappaFindWs.GetProperty( "UseAllReps4Ws" ).ToBool() );
			funResolveDisplayPatRef.SetProperty( "RefObjName", funGrappaFindWs.GetProperty( "RefObjName" ).ToString() );

		#endif

	} // if ( funGrappaFunctor.IsOk() )

	else
	{
		// reconnect roft functor to repair link broken by moving onlinetse functor above
		// from VE11A there is a new functor 'Bookkeeping4ElliScanning' to consider at this point

		// look for Bookkeeping4ElliScanning functor

		MrFunctor funBookkeepingForElliFunctor = toolbox.FindFunctor( "Bookkeeping4ElliScanning", false );

		// connect RoFt functor to Bookkeeping4ElliScanning functor if it exists, 
		// otherwise connect to rawobjprovider (as in pre-VE11A software versions)

		if ( funBookkeepingForElliFunctor.IsOk() )
		{		
			funRoFt.ConnectOutput( "ScanReady", "Bookkeeping4ElliScanning", "ComputeScan" );
		}

		else
		{
			funRoFt.ConnectOutput( "ScanReady", "rawobjprovider", "ComputeScan" );
		}

	} // else

	//----------------------------------------------------------------------------
	// ResolveMSRO Functor
	//----------------------------------------------------------------------------

	// find pipe service for ResolveObjProvider functor

	MrPipeService pipeResolveObjProvider = toolbox.FindPipeService(funResolveObjProvider.PipeServiceName().c_str());

	// create an instance of ResolveMSRO functor in the ACDecorator pipesevice

	char tResolveMSRO[64];
	sprintf( tResolveMSRO, "ResolveMSRO@%s", ICE_DLL_NAME );

	MrFunctor funResolveMSRO = pipeResolveObjProvider.CreateFunctor( "ResolveMSRO", tResolveMSRO );

	if ( !funResolveMSRO.IsOk() )
	{
		ICE_RET_ERR( "Failed to create funResolveMSRO, aborting ... ", I_FAIL );
	}

	// insert ResolveMSRO functor into preconfigured path

	toolbox.InsertAfter( "ResolveMSRO", "ComputeImage", "ImageReady", "ResolveObjProvider", "ImageReady" );
        
	// create new multi-threaded pipe service for ResolveMSRO functor

	MrPipeService pipeResolveMSRO = toolbox.SplitAfter( "ResolveObjProvider", "ResolveMSRO" );
	pipeResolveMSRO.PoolThreads( iThreadsPerThreadPool );

	// configure ResolveMSRO

	funResolveMSRO.SetProperty( "RawCols", nRawColsOS2 );
	funResolveMSRO.SetProperty( "RawLines", nRawLinesAfterGrappa );
	funResolveMSRO.SetProperty( "AdditionalCols", nAddColsOS2 );
	funResolveMSRO.SetProperty( "ROSegments", (int32_t) lROSegments );
	funResolveMSRO.SetProperty( "Shots", nShots );
	funResolveMSRO.SetProperty( "ScaleFactor", 0.0035 ); // empirical value

	//----------------------------------------------------------------------------
	// ResolveCleanUp Functor
	//----------------------------------------------------------------------------

	// find MiniHeadFillDecorator as anchor

	MrFunctor funMiniHeadFill = toolbox.FindFunctor( "MiniHeadFillDecorator" );

	// clone pipe service

	MrPipeService pipeMiniHeadFill = toolbox.FindPipeService( funMiniHeadFill.PipeServiceName().c_str() );

	// create an instance of ResolveCleanUp functor in the MiniHeadFillDecorator pipe sevice

	char tResolveCleanUp[64];
	sprintf( tResolveCleanUp, "ResolveCleanUp@%s", ICE_DLL_NAME );

	MrFunctor funResolveCleanUp = pipeMiniHeadFill.CreateFunctor( "ResolveCleanUp", tResolveCleanUp );

	if ( !funResolveCleanUp.IsOk() )
	{
		ICE_RET_ERR( "Failed to create funResolveCleanUp, aborting ... ", I_FAIL );
	}

	// insert ResolveCleanUp functor into preconfigured path

	toolbox.InsertBefore( "ResolveCleanUp", "ComputeImage", "ImageReady", "MiniHeadFillDecorator", "ComputeImage" );

	//----------------------------------------------------------------------------
	// Remove PaftFilter Functor
	//----------------------------------------------------------------------------

	// find paftfilter

	MrFunctor funPaftfilterFunctor = toolbox.FindFunctor("paftfilter", false);

	// remove paftfilter functor if it is in functor chain

	if( funPaftfilterFunctor.IsOk() )
	{
		// delete paftfilter functor

		if ( failed( toolbox.Remove("paftfilter", false) ) )
		{
			ICE_RET_ERR("Failed to remove paftfilter from functor chain, aborting...\n", I_FAIL);
		}
	}

    //----------------------------------------------------------------------------
    // Insert ResolveScanTimeDecorator
    //----------------------------------------------------------------------------

	// create ResolveScanTimeDecorator in the MiniHeadFillDecorator pipe sevice

	char tResolveScanTime[64];
	sprintf( tResolveScanTime, "ResolveScanTimeDecorator@%s", ICE_DLL_NAME );

	MrFunctor funResolveScanTime = pipeMiniHeadFill.CreateFunctor("ResolveScanTime", tResolveScanTime );

	if ( !funResolveScanTime.IsOk() )
	{
		ICE_RET_ERR( "Failed to create funResolveScanTime, aborting...\n", I_FAIL );
	}

	// insert ResolveScanTimeDecorator into preconfigured path

	toolbox.InsertAfter( "ResolveScanTime", "ComputeImage", "ImageReady", "MiniHeadFillDecorator", "ImageReady" );

	//----------------------------------------------------------------------------
	// Use special ACCombine settings for diffusion (CHARM 354957)
	//----------------------------------------------------------------------------
	MrFunctor ACDecoratorFunctor = toolbox.FindFunctor("ACDecorator", false); // doNotThrowException 

	MrPipeService pipeACDecorator;

	if( ACDecoratorFunctor.IsOk() )
	{
		pipeACDecorator = toolbox.FindPipeService(ACDecoratorFunctor.PipeServiceName().c_str());
		pipeACDecorator.PoolThreads(1);
	    
		ACDecoratorFunctor.SetProperty("bNoPC", false);
		ACDecoratorFunctor.SetProperty("bUseFirstMapOnly", true);
	}

	//----------------------------------------------------------------------------
	// Use special CoilCombineFunctor settings for diffusion (from VE11A)
	//----------------------------------------------------------------------------
	MrFunctor CoilCombineFunctor = toolbox.FindFunctor( "CoilCombineFunctor", false ); // doNotThrowException 

	if( CoilCombineFunctor.IsOk() )
	{
		MrPipeService pipeService = toolbox.FindPipeService(CoilCombineFunctor.PipeServiceName().c_str());
		pipeService.PoolThreads(1);
	    
		CoilCombineFunctor.SetProperty( "bUseFirstMapOnly", true );
	}

	//----------------------------------------------------------------------------
	// Replace standard POCS functor with ResolvePOCS functor
	//----------------------------------------------------------------------------	
	if ( ( ucROPartialFourierFactor !=  SEQ::PF_OFF ) && ( nShots < lROSegments ) )
	{
		// find standard POCSFunctor
	
		MrFunctor funPOCSFunctor = toolbox.FindFunctor( "POCSFunctor", false ); // doNotThrowException 

		if( !funPOCSFunctor.IsOk() )
		{
			ICE_RET_ERR( "Failed to find POCSFunctor pre-configured functor chain, aborting ... ", I_FAIL );
		}

		MrPipeService pipePOCSFunctor = toolbox.FindPipeService( funPOCSFunctor.PipeServiceName().c_str() );

		// create an instance of ResolvePOCSFunctor in the pipesevice containing the standard POCSFunctor

		char tResolvePOCS[64];
		sprintf( tResolvePOCS, "ResolvePOCSFunctor@%s", ICE_DLL_NAME );

		MrFunctor funResolvePOCSFunctor = pipePOCSFunctor.CreateFunctor( "ResolvePOCSFunctor", tResolvePOCS );

		if ( !funResolvePOCSFunctor.IsOk() )
		{
			ICE_RET_ERR( "Failed to create ResolvePOCSFunctor, aborting ... ", I_FAIL );
		}

		// replace standard POCS Functor with Resolve POCS functor

		toolbox.Replace( "POCSFunctor", "ResolvePOCSFunctor" );

		// manual configuration of ResolvePOCSFunctor

        int32_t nEcoCol = (int32_t) ( lBaseResolution / 2 );
        int32_t nColsPerSegment = ( nRawColsOS2 - nAddColsOS2 )/ 2;
        int32_t nCopyLenCol = nEcoCol + (int32_t) ( ( nColsPerSegment / 2 ) + ( nShots - 1 - ( lROSegments / 2 ) ) * nColsPerSegment );

        ICE_OUT("RESOLVE INFO: configuring POCS recon");
        ICE_OUT_PARAM( nEcoCol );
        ICE_OUT_PARAM( nCopyLenCol );

		funResolvePOCSFunctor.SetProperty( "POCSRO", true );
		funResolvePOCSFunctor.SetProperty( "POCSPE", false );				
		funResolvePOCSFunctor.SetProperty( "POCS3D", false );
		funResolvePOCSFunctor.SetProperty( "FirstFourierCol", 0 );
		funResolvePOCSFunctor.SetProperty( "EcoCol", nEcoCol );				
		funResolvePOCSFunctor.SetProperty( "CopyLenCol", nCopyLenCol );
		funResolvePOCSFunctor.SetProperty( "Iterations", 5 );
	}

	//----------------------------------------------------------------------------
	// Single-slice recon for debugging
	//----------------------------------------------------------------------------
	#ifdef CTRL_SliceNumber

		funRaw.SetProperty("RawSlc", 1 );

	#endif

	//----------------------------------------------------------------------------
	// User configuration from IceConfig.evp
	//----------------------------------------------------------------------------
	if( prot.At("ICE.CONFIG.Resolve.ControlMask").IsOk() )
	{
		// read control mask

		const int iConfigControlMsk = prot["ICE.CONFIG.Resolve.ControlMask"].ToLong();

        // activate tracing of absolute frequency drift during scan

        if ( iConfigControlMsk & RESOLVE_ICE_CTRL_TRACE_FREQ_DRIFT )	
        {
            // Message

            ICE_OUT( "RESOLVE INFO: tracing of absolute frequency drift has been activated in ICE.CONFIG" );

            // activate trace in ResolveFeedback functor

            funResolveFeedback.SetProperty( "TraceFreqOffsetAbs", true );
        }

		// switch off image recon

		if ( iConfigControlMsk & RESOLVE_ICE_CTRL_IMAGE_RECON_OFF )	
		{
			// Message

			ICE_OUT( "RESOLVE INFO: image recon has been switched off in ICE.CONFIG" );

			// set image recon status in ResolveGateway functor

			funResolveGateway.SetProperty("ImageRecon", false );    
		}

		// configure single-slice recon

		if ( iConfigControlMsk & RESOLVE_ICE_CTRL_SINGLE_SLICE )	
		{
			// Message

			ICE_OUT( "RESOLVE INFO: single-slice recon has been activated in ICE.CONFIG" );

			// activate single-slice recon in ResolveGateway functor

			funResolveGateway.SetProperty("SingleSliceRecon", true );    

			// specify number of slice to be reconstructed

			if( prot.At("ICE.CONFIG.Resolve.SliceNumber").IsOk() )
			{
				const int iSliceNumber = prot["ICE.CONFIG.Resolve.SliceNumber"].ToLong();
				funResolveGateway.SetProperty("SliceNumber", iSliceNumber );  
			}

			else
			{
				funResolveGateway.SetProperty("SliceNumberForSingleSliceRecon", 0 );    
			}

			// patch size of raw data object

			funRaw.SetProperty("RawSlc", 1 );
		}
	}

	//----------------------------------------------------------------------------
	// Avoid compiler warnings:
	// Reference to unused variables, that are in code for possible future use
	//----------------------------------------------------------------------------
	if (false )
	{
		ICE_OUT( nDiffusionMode );
	}

	//----------------------------------------------------------------------------
	// Report success
	//----------------------------------------------------------------------------
	ICE_OUT( "RESOLVE INFO: ResolveConfigurator::Compose() successfully completed" ); 
	*/

	//----------------------------------------------------------------------------
	// Finish
	//----------------------------------------------------------------------------
	return I_OK;

} // ResolveConfigurator::Compose()


//================================================================================
// End of PROJECT_NAMESPACE
//================================================================================
}
