Raw Data Sender and Receiver in TWIX-like format

Purpose:
Provide raw data streaming to an external computer with TWIX-like data files.

Components
  - Functor: IceNIH_RawSendFunctor
  - IceProgram: Sets parameters
  - Data Catcher: Reference Python implementation

Description: This functor sends raw data from the MRIR to an external
computer. The data is sent in a TWIX-like format and can be used for
off-line reconstruction using retro-recon and other tools.  By
default, the functor is inserted into the product functor chain right
after the "root" functor.  During EndInit, the functor opens a socket
to a data catcher and sends the Protocol in a text format identical to
what is produced by TWIX.  For each data acquisition, the functor
sends the MDH header and data.  The functor can be configured to also
send the data on to the rest of the recon chain.

TODO: On VB17, this is almost exactly like a TWIX file.  It will go
through the Siemens recon chain.  There are still some things to sort
out on the VD line.
