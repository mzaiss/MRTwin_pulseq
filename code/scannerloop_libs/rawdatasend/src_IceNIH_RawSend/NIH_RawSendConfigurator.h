#ifndef NIH_RawSendConfigurator_h
#define NIH_RawSendConfigurator_h 1


#include "MrServers/MrVista/include/Parc/ProtocolComposer.h"

class NIH_RawSendConfigurator : public Parc::Component, public ProtocolComposer::IComposer
{
public:
    DECLARE_PARC_COMPONENT(NIH_RawSendConfigurator);

    virtual IResult Compose( ProtocolComposer::Toolbox& toolbox );
};

#endif
