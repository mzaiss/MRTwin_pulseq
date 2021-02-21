
::Location of IceConfig.evp file 
set IceConfigFile=C:\MedCom\config\Ice\IceConfig.evp

::IP of the MRI console computer (internal network, connected to the ICE)
set MRIHostIP=192.168.3.1

::Port to receive data from ICE
set MRIDirectPort=6666





:: Add the Functor 
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p ActivationMode -v 1
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p AnchorsFunctorName -v "root"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p AnchorsInsertAfterEvent -v "ScanReady"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p AnchorsInsertBeforeSink -v "ComputeScan"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p NewFunctorsClassAtDllName -v "NIH_RawSendFunctor@IceNIH_RawSend"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p NewFunctorsEvent -v "ScanReady"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p NewFunctorsSink -v "ComputeScan"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property0Name -v "Hostname"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property0Value -v "%MRIHostIP%"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property1Name -v "Port"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property1Value -v "%MRIDirectPort%"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property2Name -v "BlockData"
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property2Value -v "false"




xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property3Name -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property3Value -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property3ValueFromProtocol -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property4Name -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property4Value -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property4ValueFromProtocol -v "" 
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property5Name -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property5Value -v ""
xedit -f %IceConfigFile% -n ICE.DEBUG.InsertArbitraryFunctor -p Property5ValueFromProtocol -v "" 
