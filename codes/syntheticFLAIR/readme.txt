Hi Moritz

Brain phantom with T1, T2 and proton density maps are here, along with a script that calculates the synthetic FLAIR:
https://owncloud.tuebingen.mpg.de/index.php/s/FwBLKrideXaBzti
The one MAT file includes effective voxel T1, T2, PD values, and the other includes tissue partial volume maps and the T1/T2/PD simulated for each tissue.

The general FLAIR equation is: M0*(1-2*exp(-TI/t1)) + exp(-TR/T1) )*exp(-TE/t2));   
This assumes T2 weighting, as is standard in the clinic.
As a guideline for FLAIR we use TE ~ t2WM and TI ~ -t1CSF*log(0.5), and TR long.

For example if you change the TI and run
for TI=2500:100:3500, [FLAIR] = fn_syntheticFLAIR([TI TR TE],t1im,t2im,m0im,T1T2species_true,index,1,1); end
you will see the effect of changing the inversion time.  You will notice here that there is a bright rim at the very edge of the brain; this is a partial volume effect from PV of CSF and GM.

Better is to assume the PD comes only from GM and WM. This better represents saturation of the CSF contribution :
m0im_CSFrem = T1T2species_true(2,3)*GMtrue + T1T2species_true(3,3)*WMtrue;
for TI=2500:100:3500, [FLAIR] = fn_syntheticFLAIR([TI TR TE],t1im,t2im,m0im_CSFrem,T1T2species_true,index,1,1); end
Even better is to use voxel T1 and T2 values that only reflect GM/WM mixtures.
See: Deshmane et al. Proc ISMRM 2016 p 1909.  https://index.mirasmart.com/ISMRM2016/PDFfiles/1909.html
There is a second MATLAB script in the ownCloud folder that does the whole MRF partial volume correction, if you are curious.

Hopefully this gets you quickly set up. Let me know if you need anything else!
Anagha





From: Moritz Zaiss <Moritz.Zaiss@tuebingen.mpg.de> 
Sent: Monday, 11 February, 2019 13:08
To: 'Anagha Deshmane' <Anagha.Deshmane@tuebingen.mpg.de>
Subject: in silico brain phantom

Hey Anagha,

Can you send me the in silico brain phantom data we talked about?
And if you have FLAIR signal equations in matlab that would be as well great.

Best,
Moritz


Max-Planck-Institut für biologische Kybernetik
Max-Planck-Ring 11 
72076 Tübingen 
Tel: 07071 601-735
https://www.kyb.tuebingen.mpg.de/person/59062/251691
Moritz.Zaiss@tuebingen.mpg.de

