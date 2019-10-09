
//-------------------------------------------------------------------------------------
//  Copyright (C) Fraunhofer MEVIS 2014 All Rights Reserved. Confidential
//-------------------------------------------------------------------------------------
//
//     Project: NUMARIS/4
//        File: \n4\pkg\MrServers\MrVista\Ice\IceApplicationFunctors\IceResolve\DataExportObjectMap.cpp
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

// dllInterface
#include "MrServers/MrVista/Ice/IceIdeaFunctors/dllInterface.h"

// OBJECT_MAP macro definitions
#include "MrServers/MrVista/include/Parc/Reflection/ObjectMap.h"

// Configurator
#include "DataExportConfigurator.h"

// DataExportFunctor
#include "DataExportFunctor.h"

// Export of classes outside of this Dll (needed once per Dll):
BEGIN_OBJECT_MAP()
	OBJECT_ENTRY( DataExportConfigurator )
	OBJECT_ENTRY( DataExportFunctor      )
END_OBJECT_MAP()
