sEMG.zip: the acquired raw sEMG signals, please unzip it to obtain the data.
Filters.mat: the filters in this work, including three 8-order band pass filters (F1,15-100;F2,100-550;F3,15-550) and a 2-order notch filter (F4,50Hz)
GetDataAndSpikes.m: The proposed SFD encoder.
Iterate.m: the core function to calculate as LIF model. 
Iterate_mex.mexw64: The Iterate is pre-mex as Iterate_mex.mexw64 to speed up the calculation.
LoadMeToMexTheIterate.mat: If the Iterate_mex.mexw64 does not work, please load the LoadMeToMexTheIterate.mat to mex the Iterate.m.
ResultsProcess.m: Process the obtained results.
SNNClassify.m: the main function for recognize the sEMG by SNN.

The order of usage:
0. Unzip the sEMG.zip, then get the data.
1. Run GetDataAndSpikes.m, then get the SFDSpike.mat
2. Run SNNClassify.m, then get the results.mat
3. Run ResultsProcess.m, then get the visualized results.

Note:
1. If possible, please run the SNNClassify.m at a PC with more than 10 physical cores.
2. If you need to mex the Iterate.m, please use the Matlab Coder APP with this command in step 2 "Define".
    [TrainLossTmp,~]=Iterate(SimNum,EInput,HN,ON,dVGain,WeightEH,WeightHO,SpikeThres,DropMask,DropScale,RestPotential,ac,1);
    After that, please generate mex file rather than c code.

If you have any question, please contact: yikangyang@mail.nankai.edu.cn