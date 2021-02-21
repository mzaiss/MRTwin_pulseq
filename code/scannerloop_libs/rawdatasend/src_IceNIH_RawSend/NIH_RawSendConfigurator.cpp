// MODIFICATION HISTORY:
//    2012/09/25-27, Jacco A. de Zwart, NIH
//        An option is added to allow functor instertion in a place other than
//        after 'root', which remains the default.
//    2013/09/27, JAdZ
//        Added support for local storage to disk.
//    2013/10/22, JAdZ
//        Added support for protocol anonymization.



#include "NIH_RawSendConfigurator.h"
#include "NIH_RawSendFunctor.h"
#include "MrServers/MrVista/include/Parc/Trace/IceTrace.h"

// SeqDefines
#include "MrServers/MrProtSrv/MrProt/SeqDefines.h"

using namespace MED_MRES_XProtocol;  //namespace of everything exported by XProtocol component

IResult NIH_RawSendConfigurator::Compose( ProtocolComposer::Toolbox& toolbox )
{
    using namespace ProtocolComposer;
	
    toolbox.Init( "IRIS" );

    XProtocol& prot = toolbox.Protocol();
	
    // Set properties defined in config file
	std::string sHostname = prot["NIH.RawSendFunctor.Host"].ToString();
	ICE_OUT("NIH_RawSendFunctorConfigurator set Hostname to ... " << sHostname);

	int         iPort     = prot["NIH.RawSendFunctor.Port"].ToLong();
	ICE_OUT("NIH_RawSendFunctorConfigurator set Port to ... " << iPort);

	bool        bBlock    = prot["NIH.RawSendFunctor.BlockData"].ToBool();
	ICE_OUT("NIH_RawSendFunctorConfigurator set BlockData to ... " << bBlock);

	bool        bAnonymize;
	try {
		bAnonymize = prot["NIH.RawSendFunctor.Anonymize"].ToBool();
	}
	catch (...) {
		ICE_OUT("WARNING: NIH.RawSendFunctor.Anonymize DOES NOT EXIST! Defaulting to false");
		bAnonymize = false;
	}
	ICE_OUT("NIH_RawSendFunctorConfigurator set Anonymize to ... " << bAnonymize);

	const char *sInsertFunctorAfter;
	try {
		sInsertFunctorAfter = prot["NIH.RawSendFunctor.InsertFunctorAfter"].ToString();
	}
 	catch (...) {
		#ifdef VB
			ICE_OUT("WARNING: NIH.RawSendFunctor.InsertFunctorAfter DOES NOT EXIST! Defaulting to root");
			sInsertFunctorAfter = "root";
		#else
			ICE_OUT("WARNING: NIH.RawSendFunctor.InsertFunctorAfter DOES NOT EXIST! Defaulting to RootFunctor");
			sInsertFunctorAfter = "RootFunctor";
		#endif
	}
	ICE_OUT("NIH_RawSendFunctorConfigurator set sInsertFunctorAfter to ... " << sInsertFunctorAfter);

	//	Get other property values
	int iMeasUID;
	try {
		iMeasUID = prot["HEADER.MeasUID"].ToLong();
	}
 	catch (...) {
		ICE_OUT("WARNING: Cannot find parameter HEADER.MeasUID");
	}
	ICE_OUT("NIH_RawSendFunctorConfigurator: MeasUID = " << iMeasUID);
	std::string sProtocolName;
	try {
		sProtocolName = prot["HEADER.tProtocolName"].ToString();
	}
	catch (...) {
		ICE_OUT("WARNING: Cannot find parameter HEADER.tProtocolName");
	}
	ICE_OUT("NIH_RawSendFunctorConfigurator: tProtocolName = " << sProtocolName);

	//	Prepare the functor
	MrFunctor		rootFunctor = toolbox.FindFunctor(sInsertFunctorAfter);
    std::string		pipeName = rootFunctor.PipeServiceName();
    MrPipeService	pipeService = toolbox.FindPipeService(pipeName.c_str());
    MrFunctor		NIH_RawSendFunctor = pipeService.CreateFunctor("NIH_RawSend", "NIH_RawSendFunctor@IceNIH_RawSend");

	//	Set properties
	NIH_RawSendFunctor.SetProperty("Hostname", sHostname);
	NIH_RawSendFunctor.SetProperty("Port", iPort);
	NIH_RawSendFunctor.SetProperty("BlockData", bBlock);
	NIH_RawSendFunctor.SetProperty("Anonymize", bAnonymize);
	NIH_RawSendFunctor.SetProperty("InsertFunctorAfter", sInsertFunctorAfter);
	NIH_RawSendFunctor.SetProperty("MeasUID", iMeasUID);
	NIH_RawSendFunctor.SetProperty("ProtocolName", sProtocolName);

	//  Insert the functor
	if( failed( toolbox.InsertAfter("NIH_RawSend",
								   "ComputeScan",
									"ScanReady",
									sInsertFunctorAfter, 
									"ScanReady") ) )
	{
		ICE_ERR("Failed during call of InsertAfter(NIH_RawSend->'" << sInsertFunctorAfter << "'), aborting...\n");
		return I_FAIL;
	}

    ICE_OUT( "NIH_RawSendFunctor created and inserted into functor chain." );

    return I_OK;
}
