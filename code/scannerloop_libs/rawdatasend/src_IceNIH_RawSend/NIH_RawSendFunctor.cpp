// MODIFICATION HISTORY:
//    2012/09/19, Jacco A. de Zwart, NIH
//        Added VD11 compatibility. Should still work under VB17 if the
//        environment variable VB is defined.
//    2013/09/27, JAdZ
//        Support for writing to local USB disk.
//    2013/10/22, JAdZ
//        Support for on-the-fly header anonymization.

#include <stdio.h>
#include "NIH_RawSendFunctor.h"

// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------
NIH_RawSendFunctor::NIH_RawSendFunctor()
	:	m_iPort(0),
		m_bBlockData(false),
		m_bAnonymize(false),
		m_iMeasUID(-1),
		m_sProtocolName("_unknown_"),
		m_sCopyMDH("_blank_"),
		m_bCopyMDHFlag(true)
{
		// Register call backs
		addCB(IFirstCall);
}

// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------
NIH_RawSendFunctor::~NIH_RawSendFunctor()
{
}

// --------------------------------------------------------------------------
// EndInit
// --------------------------------------------------------------------------
IResult NIH_RawSendFunctor::EndInit(IParcEnvironment* env )
{
	ICE_SET_FN("NIH_RawSendFunctor::EndInit()")
	ICE_OUT("NIH_RawSendFunctor::EndInit");

	// call base functionality
	if(failed(IceScanFunctors::EndInit(env)))
	{
		ICE_RET_ERR("Failed during call of IceScanFunctors::EndInit(), aborting...\n", I_FAIL);
	}

	// by default this is socket based
	localtarget=0;

	// List the contents of the PARC. This is informative. 
	env->ListContent();

	// Get the buffer containing the protocols from the PARC
	IAddBuffer::Pointer pProtData = Parc::cast<IAddBuffer>(env->Item("addBuffer"));	
	size_t avail  = pProtData->getBufLen();
	char *pProts = new char [avail];
	size_t written = 0;
	written = pProtData->write(pProts, avail);
	if (written != avail) {
		ICE_OUT("NIH_RawSendFunctor: Error retrieving protoocols.");
		return I_FAIL;
	}
	// Unfortunately on the simulator we are missing a few of the buffers.
	#ifndef _WINDLL
		char			command[256],line[256];
	#endif

	// if hostname is "localhost" probe the first partition on the first USB disk for compliance
	if (!strcmp(m_sHostname.c_str(),"localhost")) {

		// define localtarget
		localtarget=1;

		// this should only run on the linux-based MRIR
		#ifndef _WINDLL

			// local variables
			struct stat		st;
    		int				status=0;
			time_t			now;
 			struct tm		timestruct;

			// step 1: create /nih if it does not exist
    		if (stat("/nih",&st) != 0) {
        		if (mkdir("/nih",S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH) != 0) {
					status=1;
				} else ICE_OUT("NIH_RawSendFunctor: /nih created!");
			} else if (!S_ISDIR(st.st_mode)) status=1;
			if (status == 1) {
				ICE_OUT("NIH_RawSendFunctor: Unable to create /nih, quitting!");
				return(I_FAIL);
			}

			// step 2: If a file /nih/.NIH_RawSend_datadisk does not exist try to mount the first partition on the first USB disk
			if (stat("/nih/.NIH_RawSend_datadisk",&st) != 0) {

				// step 2.1: is a USB disk connected? if not quit
				#ifdef VB
					strcpy(command,"udevinfo --export-db");
				#else
					strcpy(command,"udevadm info --export-db");
				#endif
				strcat(command," | grep usb | awk 'BEGIN{ DEV=\"\"; }; /block/ { split($0,JUNK,\"/block/\"); if (DEV == \"\") { DEV=JUNK[2] }; }; END { print(DEV) };'");
				fp=popen(command,"r");
				if (fgets(line,PATH_MAX,fp) == NULL) {
					ICE_OUT("NIH_RawSendFunctor: Unable to probe for USB storage device, quitting!");
					return(I_FAIL);
				}
				pclose(fp);

				// step 2.2: Try to mount the first partition on that disk on /nih/
				strcpy(command,"mount /dev/");
				strncat(command,line,3);			// copy only the first three characters, e.g. sdd, since there is a newline char as well
				strcat(command,"1 /nih ; df -h /nih");
				fp=popen(command,"r");
				if (fgets(line,PATH_MAX,fp) == NULL) {
					ICE_OUT("NIH_RawSendFunctor: Unable to mount device on /nih, quitting!");
					return(I_FAIL);
				}
				pclose(fp);

				// step 2.3: Quit if the file /nih/.NIH_RawSend_datadisk still does not exist
				if (stat("/nih/.NIH_RawSend_datadisk",&st) != 0) {
					ICE_OUT("NIH_RawSendFunctor: Unable to find /nih/.NIH_RawSend_datadisk, quitting!");
					return(I_FAIL);
				}
			}

			// step 3: does a data/ folder exist? if not create it with 777 permission mode
    		if (stat("/nih/data",&st) != 0) {
        		if (mkdir("/nih/data",S_IRWXU|S_IRWXG|S_IRWXO) != 0) status=1;
				if (!S_ISDIR(st.st_mode)) status=1;
				if (status == 1) {
					ICE_OUT("NIH_RawSendFunctor: Unable to create /nih/data, quitting!");
					return(I_FAIL);
				}
			}

			// step 4: open a file
			time(&now);
			timestruct=*localtime(&now);
			strftime(datafile,sizeof(datafile),"/nih/data/meas_MID%H%M%S_%Y%m%d.dat",&timestruct);
			fp=fopen(datafile,"w+");
			if (fp == NULL) {
				ICE_OUT("NIH_RawSendFunctor: Unable to open file " << datafile << " for writing, quitting!");
				return(I_FAIL);
			} else ICE_OUT("NIH_RawSendFunctor: Opened output file " << datafile);

		// end of linux only code
		#endif

	// otherwise set up a socket connection
	} else {

		// Server Address and Port
		ACE_INET_Addr addr ((u_short) m_iPort, m_sHostname.c_str());
 
		// Connect to the server
		if (connector.connect (server, addr) == -1) {
			ICE_OUT("NIH_RawSendFunctor: Failed to connect to the raw server.");
			return I_FAIL;
		}

		ICE_OUT("NIH_RawSendFunctor: Connected to the raw server.");
	}

	//	if this is VD, fill and send the multi-file wrapper
	#ifdef VD

		//	Print info
		ICE_OUT("NIH_RawSendFunctor: This is a post-VD11A system - send a data set wrapper of " << VD_WRAPPER_BYTES << " bytes.");

		//	variable definition
		int i;
		unsigned char *pWrapper = new unsigned char [VD_WRAPPER_BYTES];
		uint32_t *pulong = (uint32_t *)pWrapper;
		uint64_t *pulong64 = (uint64_t *)pWrapper;

		//	zero-fill the wrapper - for some unknown reason bytes 32-51 are set to 'x' aka byte 120
		for (i=0; i<VD_WRAPPER_BYTES; i++) {
			if ((i > 31) && (i < 52)) pWrapper[i]=120; else 
				pWrapper[i]=0;
		}

		//	define that this file only contains a single set
		pulong[1]=1;

		//	set the MeasUID and FIDnumber
		pulong[2]=m_iMeasUID;
		//pulong[3]=m_iFIDnr;

		//	set the offset to the first set
		pulong64[2]=VD_WRAPPER_BYTES;

		//	set the protocol name
		const unsigned char *cprot = (const unsigned char *)m_sProtocolName.c_str();
		for (i=0; i<(int)m_sProtocolName.length(); i++) pWrapper[i+96]=cprot[i];

		// Send the wrapper
		if (localtarget == 1) {
			written=fwrite(pWrapper,VD_WRAPPER_BYTES,1,fp);
			if (written == 0) {
				ICE_OUT("NIH_RawSendFunctor: Failed to write VD wrapper.");
				return(I_FAIL);
			}
		} else server.send_n(pWrapper,VD_WRAPPER_BYTES);
		ICE_OUT("NIH_RawSendFunctor: Sent VD-style wrapper.");
	#endif

	// Send the protocols retrieved from PARC
	if (localtarget == 1) {
		#ifndef _WINDLL
			written=fwrite(pProts,avail,1,fp);
			if (written == 0) {
				ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
				return(I_FAIL);
			}
		#endif
	} else server.send_n(pProts, avail);

	ICE_OUT("NIH_RawSendFunctor: Sent text protocol.");

	// Clean up
	delete[] pProts;

	return I_OK;
}

// --------------------------------------------------------------------------
// FirstCall
//   This method is called once if the functor is called for the first time
// --------------------------------------------------------------------------
IResult NIH_RawSendFunctor::FirstCall( IceAs& srcAs, MdhProxy& aMdh, ScanControl& ctrl )
{

    ICE_SET_FN("NIH_RawSendFunctor::FirstCall()")

	//	Force copying of MDH to string
	m_bCopyMDHFlag=true;

	// Some debugging
	ICE_OUT("NIH_RawSendFunctor::FirstCall");

	return I_OK;
}

// --------------------------------------------------------------------------
// ComputeScan
// --------------------------------------------------------------------------
IResult NIH_RawSendFunctor::ComputeScan(IceAs& srcAs, MdhProxy& aMdh, ScanControl& ctrl)
{

	ICE_SET_FN("NIH_RawSendFunctor::ComputeScan()")

	// Execute call back
    IResult res = ExecuteCallbacks(srcAs, aMdh, ctrl);
    if(failed(res)) {
        ICE_RET_ERR("NIH_RawSendFunctor: ExecuteCallbacks failed.  Aborting...", res);
    }

	// Send the data to the catcher
	long unsigned int	dataSize;
	uint32_t			dmamask=MDH_DMA_LENGTH_MASK;
	#ifdef VB
		sMDH			m_mdh;
	#else
		sScanHeader		m_mdh;
		sChannelHeader	c_mdh;
	#endif
	#ifndef _WINDLL
		size_t			written = 0;
	#endif

	//	compute dataSize
	dataSize = 2*aMdh.getNoOfColumns()*sizeof(float);

	// Copy the current MDH
	memcpy(&m_mdh, aMdh.getMdhData(), sizeof(m_mdh));

	//	Fill in the sChannelHeader fields as best as possible since it is not
	//	available 'behind' the root functor
	#ifdef VD
		//c_mdh.ulTypeAndChannelLength=???
		c_mdh.lMeasUID=m_mdh.lMeasUID;
		c_mdh.ulScanCounter=m_mdh.ulScanCounter;
		//c_mdh.ulSequenceTime=m_mdh.ulTimeStamp;
		//c_mdh.CRC=???;
	#endif

	//	Store a copy of MDH for the EndOfJob function
	if (m_bCopyMDHFlag) {
		ICE_OUT("NIH_RawSendFunctor: Storing a copy of MDH for use in NIH_RawSendFunctor::endOfJob()");
		char			myMdh[2*sizeof(m_mdh)];
		int				hexword[2];
		int				cval;
		unsigned char	*pMdh=(unsigned char *)&m_mdh;
		for (int i=0; i<(int)sizeof(m_mdh); i++) {
			cval=(int)pMdh[i];
			hexword[0]=(cval/16)+48;
			if (hexword[0] > 57) hexword[0]+=7;
			hexword[1]=(cval % 16)+48;
			if (hexword[1] > 57) hexword[1]+=7;
			myMdh[2*i]=(char)hexword[0];
			myMdh[2*i+1]=(char)hexword[1];
		}
		m_sCopyMDH=myMdh;
		m_bCopyMDHFlag=false;
	}

	//	For VD send one MDH before the data loop
	#ifdef VD
		if (localtarget == 1) {
			#ifndef _WINDLL
				written=fwrite(&m_mdh,sizeof(m_mdh),1,fp);
				if (written == 0) {
					ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
					return(I_FAIL);
				}
			#endif
		} server.send_n(&m_mdh, sizeof(m_mdh));	
	#endif

	//	Loop over channels
	for (int c=0; c < aMdh.getNoOfChannels(); c++) {

		//	for VB, send an MDH for each channel, for VD send a channel header instead
		#ifdef VB
			//	For channels >=1 the dma length appears to be half of the
			//	channel 0 value (apply mask to get correct bits see
			//	MDH_DMA_LENGTH_MASK in mdh.h)
			if (c == 1) m_mdh.ulFlagsAndDMALength-=((m_mdh.ulFlagsAndDMALength & dmamask)/2);

			//	set the channel number, which seems to be consecutive for VB
			m_mdh.ushChannelId = static_cast<unsigned short>(c);
			if (localtarget == 1) {
				#ifndef _WINDLL
					written=fwrite(&m_mdh,sizeof(m_mdh),1,fp);
					if (written == 0) {
						ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
						return(I_FAIL);
					}
				#endif
			} server.send_n(&m_mdh, sizeof(m_mdh));
		#else

			// Copy the current channel data, not sure if this is correct...
			c_mdh.ulChannelId = static_cast<unsigned short>(c);
			if (localtarget == 1) {
				#ifndef _WINDLL
					written=fwrite(&c_mdh,sizeof(c_mdh),1,fp);
					if (written == 0) {
						ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
						return(I_FAIL);
					}
				#endif
			} server.send_n(&c_mdh, sizeof(c_mdh));
		#endif

		//	Send Data
		if (!srcAs.modify(CHA, c, 1, 1))
			ICE_RET_ERR("NIH_RawSendFunctor: Modify of srcAs failed", I_FAIL);
			if (localtarget == 1) {
				#ifndef _WINDLL
					written=fwrite(srcAs.calcSplObjStartAddr(),dataSize,1,fp);
					if (written == 0) {
						ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
						return(I_FAIL);
					}
				#endif
			} server.send_n(srcAs.calcSplObjStartAddr(), dataSize);
	}

	// If NOT blocking, pass the data along to the next functor
	if (m_bBlockData == false) {
		// Reset the access specifier to the right number of channels
		if (!srcAs.modify(CHA, 0, aMdh.getNoOfChannels(), 1))
			ICE_RET_ERR("NIH_RawSendFunctor: Reset of srcAs failed", I_FAIL);
		// Send
		ScanReady(srcAs, aMdh, ctrl);
	}

	return I_OK;
}

// --------------------------------------------------------------------------
// endOfJob
// --------------------------------------------------------------------------
IResult NIH_RawSendFunctor::endOfJob(IResult reason)
{	

	ICE_SET_FN("NIH_RawSendFunctor::endOfJob()")
	ICE_OUT("NIH_RawSendFunctor::endOfJob()");

	if (reason == I_ACQ_END) {
		// Acquisition finished - write an MDH_ACQEND event

		#ifdef VB
			sMDH			m_mdh;
		#else
			sScanHeader		m_mdh;
			sChannelHeader	c_mdh;
		#endif
		const unsigned char	*myMdh=(const unsigned char *)m_sCopyMDH.c_str();
		int					hexword[2];
		int					cval;
		unsigned char		*pMdh=(unsigned char *)&m_mdh;
		int					c,i;
		uint32_t			dmamask=MDH_DMA_LENGTH_MASK;
		#ifndef _WINDLL
			size_t			written = 0;
		#endif

		//	retrieve Mdh from m_sCopyMDH
		ICE_OUT("NIH_RawSendFunctor: retrieving MDH from m_sCopyMDH");
		for (i=0; i<(int)sizeof(m_mdh); i++) {
			hexword[0]=(int)(myMdh[2*i])-48;
			if (hexword[0] > 15) hexword[0]-=7;
			hexword[1]=(int)(myMdh[2*i+1])-48;
			if (hexword[1] > 15) hexword[1]-=7;
			cval=16*hexword[0]+hexword[1];
			pMdh[i]=(unsigned char)cval;
		}

		//	override some MDH values
		m_mdh.ulFlagsAndDMALength=3221233664;		//	= 8192 when masked with MDH_DMA_LENGTH_MASK
		m_mdh.aulEvalInfoMask[0] = 1L;
		m_mdh.aulEvalInfoMask[1] = 0L;

		//	under VD send one mdh
		#ifdef VD
			m_mdh.ushUsedChannels = 0;
			m_mdh.ushSamplesInScan = 0;
			if (localtarget == 1) {
				#ifndef _WINDLL
					written=fwrite(&m_mdh,sizeof(m_mdh),1,fp);
					if (written == 0) {
						ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
						return(I_FAIL);
					}
				#endif
			} server.send_n(&m_mdh, sizeof(m_mdh));

		//	under VB send one complete set with 16-sample data (128 bytes)
		#else
			m_mdh.ushSamplesInScan = 16;
			char blankdat[128];
			for (i=0; i<128; i++) blankdat[i]=0;
			for (c=0; c<m_mdh.ushUsedChannels; c++) {
				if (c == 1) m_mdh.ulFlagsAndDMALength-=((m_mdh.ulFlagsAndDMALength & dmamask)/2);
				m_mdh.ushChannelId = static_cast<unsigned short>(c);
				if (localtarget == 1) {
					#ifndef _WINDLL
						written=fwrite(&m_mdh,sizeof(m_mdh),1,fp);
						if (written == 0) {
							ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
							return(I_FAIL);
						}
						written=fwrite(&blankdat,sizeof(blankdat),1,fp);
						if (written == 0) {
							ICE_OUT("NIH_RawSendFunctor: Failed to write protocol.");
							return(I_FAIL);
						}
					#endif
				} else {
					server.send_n(&m_mdh, sizeof(m_mdh));
					server.send_n(&blankdat, sizeof(blankdat));
				}
			}
		#endif
	}
	ICE_OUT("\nNIH_RawSendFunctor::endOfJob(): Send ACQEND flag.\n");

	//	Close the server or file
	if (localtarget == 1) {
		#ifndef _WINDLL

			// local variables
			char command[256],line[256];

			// close the file
			if (fclose(fp) != 0) {
				ICE_OUT("NIH_RawSendFunctor: Failed to close the data file.");
				return(I_FAIL);
			}

			// change file permissions to 666
			chmod(datafile,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);

			// unmount the /nih disk
			strcpy(command,"umount -v /nih/");
			fp=popen(command,"r");
			if (fgets(line,PATH_MAX,fp) == NULL) {
				ICE_OUT("NIH_RawSendFunctor: Unable to umount the USB disk attached to /nih!");
				return(I_FAIL);
			}
			pclose(fp);

		#endif
	} else {
		if (server.close() == -1) {
			ICE_OUT("\nUnable to close server.\n");
			return I_FAIL;
		}
		ICE_OUT("\nNIH_RawSendFunctor::endOfJob(): Closed connection to the raw server.\n");
	}

	return I_OK;
}
