
//-------------------------------------------------------------------------------------
//  Copyright (C) Fraunhofer MEVIS 2014 All Rights Reserved. Confidential
//-------------------------------------------------------------------------------------
//
//     Project: NUMARIS/4
//        File: \n4\pkg\MrServers\MrVista\Ice\IceApplicationFunctors\IceResolve\DataExportConfigurator.h
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
// Start of DataExportConfigurator_h wrapper
//================================================================================
#ifndef DataExportConfigurator_h
#define DataExportConfigurator_h 1


//================================================================================
// Includes
//================================================================================

// ProtocolComposer
#include "MrServers/MrVista/include/Parc/ProtocolComposer.h"

// DataExportDefs
#include "DataExportDefs.h"


//================================================================================
// Start of project namespace
//================================================================================
namespace PROJECT_NAMESPACE
{

//================================================================================
// DataExportConfigurator class
//================================================================================
class DataExportConfigurator : public Parc::Component, public ProtocolComposer::IComposer
{
	public:

		DECLARE_PARC_COMPONENT(DataExportConfigurator);

		virtual IResult Compose( ProtocolComposer::Toolbox& toolbox );
};


//================================================================================
// End of project namespace
//================================================================================
}

//================================================================================
// End of DataExportConfigurator_h wrapper
//================================================================================
#endif
