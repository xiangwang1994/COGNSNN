function [Loss,Result]=Iterate(SimNum,EInput,HN,ON,dVGain,WeightEH,WeightHO,...
    SpikeThres,DropMask,DropScale,RestPotential,ac,ifDrop)
% Input spike trains
ES=EInput;
% Hidden layer input
HInput=zeros(HN,SimNum);
% Neuron membrane in the hidden layer
HV=HInput;
% Hidden layer spikes
HS=HV;
% Output layer input
OInput=zeros(ON,SimNum);
% Neuron membrane in the hidden layer
OV=OInput;
% Hidden layer
for sn=1:SimNum-1
    HInput(:,sn)=WeightEH'*ES(:,sn);
    HV(:,sn+1)=HV(:,sn)+dVGain*(HInput(:,sn)-HV(:,sn));
    HIndex=find(HV(:,sn+1)>=SpikeThres);
    if ~isempty(HIndex)
        HV(HIndex,sn+1)=RestPotential;
        HS(HIndex,sn+1)=1;
    end
end
% Dropout
if ifDrop
    HS=HS.*DropMask*DropScale;
end
% Output layer
for sn=1:SimNum-1
    OInput(:,sn)=WeightHO'*HS(:,sn);
    OV(:,sn+1)=OV(:,sn)+dVGain*(OInput(:,sn)-OV(:,sn));
end
% If the hidden layer generate no spike, give a big punish
if sum(HS,'all')==0
    Loss=-1000;
    Result=0;
else % 1e-25 is used to aviod ln(0)
    OutputEnergy=sum(OV,2)+1e-25;
    Loss=log(OutputEnergy(ac)/sum(OutputEnergy));
    [~,Result]=max(OutputEnergy);
end
end