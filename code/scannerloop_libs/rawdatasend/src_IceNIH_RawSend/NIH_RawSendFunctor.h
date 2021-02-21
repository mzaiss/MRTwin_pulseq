// MODIFICATION HISTORY:
//    2012/09/25, Jacco A. de Zwart, NIH
//        An option is added to allow functor instertion in a place other than
//        after 'root', which remains the default.


#ifndef NIHRAWSENDFUNCTOR_H
#define NIHRAWSENDFUNCTOR_H

// Import/Export DLL macro:
#include "MrServers/MrVista/include/Ice/IceScanFunctors/dllInterface.h"
// Base class IceScanFunctors
#include "MrServers/MrVista/include/Ice/IceUtils/IceScanFunctors.h"

// Protocol
#include "MrServers/MrVista/include/Parc/ProtocolComposer.h"

// Needed system includes
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

// Use the socket stuff from ACE
#include <ace/INET_Addr.h>
#include <ace/SOCK_Stream.h>
#include <ace/SOCK_Connector.h>

//	Local defines
#ifdef VD
	#define	VD_WRAPPER_BYTES 10240
// Some day this may be useful for debugging
//#include "MrServers/MrMeasSrv/SeqIF/MDH/MdhUtils.h"
#endif

class NIH_RawSendFunctor : public IceScanFunctors
{
public:
    ICE_SCAN_FUNCTOR(NIH_RawSendFunctor);

	BEGIN_PROPERTY_MAP(NIH_RawSendFunctor)
        PROPERTY_ENTRY ( Hostname )
        PROPERTY_ENTRY ( Port )
		PROPERTY_ENTRY( BlockData )
		PROPERTY_ENTRY( Anonymize )
		PROPERTY_ENTRY( InsertFunctorAfter )
		PROPERTY_ENTRY( MeasUID )
		PROPERTY_ENTRY( ProtocolName )
		PROPERTY_ENTRY( CopyMDH )
		PROPERTY_ENTRY( CopyMDHFlag )
	END_PROPERTY_MAP()
	
    DECL_GET_SET_PROPERTY(std::string, s, Hostname)
    DECL_GET_SET_PROPERTY(int, i, Port) 
	DECL_GET_SET_PROPERTY(bool, b, BlockData)
	DECL_GET_SET_PROPERTY(bool, b, Anonymize)
    DECL_GET_SET_PROPERTY(std::string, s, InsertFunctorAfter)
    DECL_GET_SET_PROPERTY(int, i, MeasUID) 
    DECL_GET_SET_PROPERTY(std::string, s, ProtocolName)
    DECL_GET_SET_PROPERTY(std::string, s, CopyMDH)
	DECL_GET_SET_PROPERTY(bool, b, CopyMDHFlag)

	// Constructor/Destructor
    NIH_RawSendFunctor();
    virtual ~NIH_RawSendFunctor();

	// Callbacks
	virtual IResult EndInit( IParcEnvironment* env );
    virtual IResult FirstCall( IceAs& srcAs, MdhProxy& aMdh, ScanControl& ctrl );

	// Event Sink
    virtual IResult ComputeScan(IceAs& srcAs, MdhProxy& aMdh, ScanControl& ctrl);
    virtual IResult endOfJob(IResult reason);

protected:
	
private:

	// ACE bits
	ACE_SOCK_Stream server;
	ACE_SOCK_Connector connector;

	// file pointer for writing to local disk
	int		localtarget;
	FILE	*fp;
	char	datafile[256];

	// function declarations
    NIH_RawSendFunctor(const NIH_RawSendFunctor &right);
    NIH_RawSendFunctor& operator=(const NIH_RawSendFunctor &right);
};

#endif
