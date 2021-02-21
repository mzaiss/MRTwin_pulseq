#include "NIH_RawSendConfigurator.h"
#include "NIH_RawSendFunctor.h"

// OBJECT_MAP macro definitions
#include "MrServers/MrVista/include/Parc/Reflection/ObjectMap.h"

//using namespace PARC_MODULE_NAME;

// Export of classes outside of this Dll (needed once per Dll):
BEGIN_OBJECT_MAP()
    OBJECT_ENTRY(NIH_RawSendConfigurator)
    OBJECT_ENTRY(NIH_RawSendFunctor)
END_OBJECT_MAP()
