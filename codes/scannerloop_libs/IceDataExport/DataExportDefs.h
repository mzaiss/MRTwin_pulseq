
//-------------------------------------------------------------------------------------
//  Copyright (C) Fraunhofer MEVIS 2014 All Rights Reserved. Confidential
//-------------------------------------------------------------------------------------
//
//     Project: NUMARIS/4
//        File: \n4\pkg\MrServers\MrVista\Ice\IceApplicationFunctors\IceResolve\DataExportDefs.h
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


//====================================================================================================
// Start of ResolveDefs_h wrapper
//====================================================================================================
#ifndef DataExportDefs_h
#define DataExportDefs_h 1


//====================================================================================================
// All sequence and ICE files including this header file use a common namespace
//====================================================================================================

// create namespace
namespace PROJECT_NAMESPACE {}

// activate namespace
using namespace PROJECT_NAMESPACE;


//====================================================================================================
// Enclose contents of this header in SEQ_NAMESPACE
//====================================================================================================
namespace PROJECT_NAMESPACE
{

//====================================================================================================
// Macros
//====================================================================================================
#include <string>  
#include <iostream>  
#include <string.h>  
#include <stdlib.h> 

using namespace std;

#define ICE_OUT_PARAM( param ) ICE_OUT( "\n********** " << #param << ": " << param << "\n" );
#define TRACE_PARAM( param ) cout << #param << ": " << param << endl;


//====================================================================================================
// Naming parameters
//====================================================================================================
//#define ICE_PROGRAM_NAME "%SiemensIceProgs%\\IceProgramResolve"
//#define SEQUENCE_STRING "resolve"
//#define EVA_FILENAME "%SiemensEvaDefProt%\\DTI\\Dti.evp"

#define ANCHOR_FUNCTOR "RootFunctor"
#define ICE_DLL_NAME "IceDataExport"

#if defined(VXWORKS) 
#define FILENAME_EXPORT "/host/MriCustomer/seq/pulseq/data/ExportedData"
#elif defined (BUILD_PLATFORM_LINUX)
#define FILENAME_EXPORT "/opt/medcom/MriCustomer/seq/pulseq/data/ExportedData"
#else
#define FILENAME_EXPORT "C:\\ExportedData"
#endif


//====================================================================================================
// Defines
//====================================================================================================


//====================================================================================================
// End of PROJECT_NAMESPACE
//====================================================================================================
}

//====================================================================================================
// End of DataExportDefs_h wrapper
//====================================================================================================
#endif


