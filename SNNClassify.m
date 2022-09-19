clear;close all;dbstop if error
% Neurons number in the hidden layer 
HN=20;
Epoch=1000;
% Windows length, ms
WL=100;
% step length
St=50;
% sEMG frequency
Frq=1500/1000;
% Gesture number
AcN=5;
% Gesture length
AcL=3000;
% Subject number
SubNum=11;
% Experiments number
ExN=10;
% Accuracy of training, testing and validating sets.
TrainAcc=cell(SubNum,ExN);
TestAcc=cell(SubNum,ExN);
ValidAcc=cell(SubNum,ExN);
% Confusion matrix of training, testing and validating sets.
TrainCM=cell(SubNum,ExN);
TestCM=cell(SubNum,ExN);
ValidCM=cell(SubNum,ExN);
% Loss
TrainScore=cell(SubNum,ExN);
ValidScore=cell(SubNum,ExN);
% Parameters of SNN
SNN=cell(SubNum,ExN);
% Recognizing results
Results=cell(SubNum,ExN);
% Epochs to get the max accuracy of validating set
MaxAccEpoch=zeros(SubNum,ExN);
% Probability of the hidden layer does not output spikes
TrainZeroRatio=cell(SubNum,ExN);
ValidZeroRatio=cell(SubNum,ExN);
% How much spike trains are treated as the validating set
ValidRatio=2/9;
% Read the spikes
load('Spikes.mat')
% Read the index
load('Index.mat')
%% Train and test
for ni=1:SubNum
    Spikes=Spike{ni};
    % Cross-validation
    parfor ex=1:ExN
        [SNN{ni,ex},TrainAcc{ni,ex},TrainCM{ni,ex},ValidAcc{ni,ex},ValidCM{ni,ex},...
            TestAcc{ni,ex},TestCM{ni,ex},Results{ni,ex},MaxAccEpoch(ni,ex),...
            TrainScore{ni,ex},ValidScore{ni,ex},TrainZeroRatio{ni,ex},ValidZeroRatio{ni,ex}]=...
            TrainSNN(Spikes,ExN,AcN,AcL,WL,St,Frq,HN,Epoch,ni,Index(ex,:),ValidRatio,ex);
    end
end
%% Save the results
save('Results.mat','SNN','TrainAcc','TrainCM','ValidAcc','ValidCM','TestAcc','TestCM',...
    'Results','TrainScore','ValidScore','MaxAccEpoch','TrainZeroRatio','ValidZeroRatio')
%% TrainSNN
function [SNN,TrainAcc,TrainCM,ValidAcc,ValidCM,TestAcc,TestCM,Results,...
    MaxAccEpoch,TrainScore,ValidScore,TrainZeroRatio,ValidZeroRatio]...
    =TrainSNN(Spike,ExN,AcN,AcL,WL,St,Frq,HN,Epoch,ni,Index,ValidRatio,fold)
% Max accuracy of validating set
MaxAcc=zeros(AcN,2);
% Epochs to get the max accuracy of validating set
MaxAccEpoch=0;
% Recognizing results
Results=cell(3,1);
% Weights of input to hidden layer, weights of hidden to output layer, spiking threshold of hidden layer
SNN=cell(3,1);
% Dropout ratio
DR=0.2;
% Windows number per gesture
WNPA=(AcL-WL)/St+1;
TrainWinIndex=Index{1};
ValidWinIndex=Index{2};
TestWinIndex=Index{3};
TotalTrainWinNum=length(TrainWinIndex);
TotalValidWinNum=length(ValidWinIndex);
TotalTestWinNum=length(TestWinIndex);
ValidWinNum=ceil(ValidRatio*WNPA);
TrainWinNum=WNPA-ValidWinNum;
TestWinNum=WNPA;
SimulateStep=1/Frq/1000;
SimNum=WL*Frq;
TrainAcc=zeros(Epoch,AcN);
ValidAcc=zeros(Epoch,AcN);
TestAcc=zeros(AcN,1);
TrainCM=zeros(Epoch,AcN,AcN);
ValidCM=zeros(Epoch,AcN,AcN);
TestCM=zeros(AcN,AcN);
% Agent number of GWO
AgentNum=150;
% neurons number in the input layer
EN=size(Spike,4);
% neurons number in the output layer
ON=AcN;
% Weights number of input to hidden layer
EHWN=EN*HN;
% Weights number of hidden to output layer
HOWN=HN*ON;
% Total number of weights
WeightNumber=EHWN+HOWN;
% Randomly initialize the parameters of SNN
Parameters=[rand(AgentNum,WeightNumber) 0.5*rand(AgentNum,1)];
% low and up limit of parameters
lb=zeros(1,length(Parameters));
ub=ones(1,length(Parameters));
% Wolf leaders
Alpha_pos=zeros(1,length(Parameters));
Alpha_score=inf; 
Beta_pos=zeros(1,length(Parameters));
Beta_score=inf; 
Delta_pos=zeros(1,length(Parameters));
Delta_score=inf;
% Loss
TrainScore=zeros(Epoch,1);
ValidScore=zeros(Epoch,1);
TrainZeroRatio=zeros(Epoch,1);
ValidZeroRatio=zeros(Epoch,1);
RestPotential=0;
% Membrane constant
Tau=0.01;
dVGain=SimulateStep/Tau;
% LIF model
% taw*v_dot=-(v-v_rest)+I_ext*R
% Calculation time
TotalTime=0;
% Expected results
ExpectTrainResults=repmat(repelem(1:AcN,1,TrainWinNum),1,ExN-1);
ExpectValidResults=repmat(repelem(1:AcN,1,ValidWinNum),1,ExN-1);
ExpectTestResults=repelem(1:AcN,1,TestWinNum);
for e=1:Epoch
    tic
    for AgentIndex=1:AgentNum
        % For every wolf, initialize the loss
        TrainLoss=0;
        % If the parameters exceed the limit, reset them to the limit
        Flag4ub=Parameters(AgentIndex,:)>ub;
        Flag4lb=Parameters(AgentIndex,:)<lb;
        Parameters(AgentIndex,:)=(Parameters(AgentIndex,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        WeightEH=reshape(Parameters(AgentIndex,1:EHWN),EN,HN);
        WeightHO=reshape(Parameters(AgentIndex,EHWN+1:EHWN+HOWN),HN,ON);
        SpikeThres=Parameters(AgentIndex,WeightNumber+1);
        % Initialize dropout
        DropMask=ones(HN,1);
        % The number of droped neurons < 4
        DropRandTmp=rand(1,HN);
        [~,DropIndex]=sort(DropRandTmp);
        DropMask(intersect(find(DropRandTmp<=DR),DropIndex(1:4)))=0;
        DropScale=HN/sum(DropMask);
        % Traversal all spike trains in the training set
        for WinIndex=1:TotalTrainWinNum
            % Get the real index
            RealIndex=TrainWinIndex(WinIndex);
            % Get the experiments, gesture and spike trains index
            ex=fix((RealIndex-1)/(AcN*WNPA))+1;
            ac=fix((RealIndex-(ex-1)*AcN*WNPA-1)/WNPA)+1;
            win=RealIndex-(ex-1)*AcN*WNPA-(ac-1)*WNPA;
            % Get spikes
            EInput=reshape([Spike{ex,ac,win,:}],SimNum,EN)';
            % SNN
            [TrainLossTmp,~]=Iterate_mex(SimNum,EInput,HN,ON,dVGain,...
                WeightEH,WeightHO,SpikeThres,DropMask,DropScale,RestPotential,ac,1);
            TrainLoss=TrainLoss-TrainLossTmp;
        end
        TrainLoss=TrainLoss/TotalTrainWinNum;
        % Update wolf leader location
        if TrainLoss<Alpha_score
            Alpha_score=TrainLoss; % Update alpha
            Alpha_pos=Parameters(AgentIndex,:);
        end
        if TrainLoss>Alpha_score && TrainLoss<Beta_score
            Beta_score=TrainLoss; % Update beta
            Beta_pos=Parameters(AgentIndex,:);
        end
        if TrainLoss>Alpha_score && TrainLoss>Beta_score && TrainLoss<Delta_score 
            Delta_score=TrainLoss; % Update delta
            Delta_pos=Parameters(AgentIndex,:);
        end
    end
    a=2-e*((2)/Epoch);
    % Update 
    for AgentIndex=1:size(Parameters,1)
        for AgentDim=1:size(Parameters,2)    
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            D_alpha=abs(C1*Alpha_pos(AgentDim)-Parameters(AgentIndex,AgentDim)); % Equation (3.5)-part 1
            X1=Alpha_pos(AgentDim)-A1*D_alpha; % Equation (3.6)-part 1
            r1=rand();
            r2=rand();
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            D_beta=abs(C2*Beta_pos(AgentDim)-Parameters(AgentIndex,AgentDim)); % Equation (3.5)-part 2
            X2=Beta_pos(AgentDim)-A2*D_beta; % Equation (3.6)-part 2
            r1=rand();
            r2=rand();
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            D_delta=abs(C3*Delta_pos(AgentDim)-Parameters(AgentIndex,AgentDim)); % Equation (3.5)-part 3
            X3=Delta_pos(AgentDim)-A3*D_delta; % Equation (3.5)-part 3
            Parameters(AgentIndex,AgentDim)=(X1+X2+X3)/3;% Equation (3.7)
        end
    end
    WeightEH=reshape(Alpha_pos(1:EHWN),EN,HN);
    WeightHO=reshape(Alpha_pos(EHWN+1:EHWN+HOWN),HN,ON);
    SpikeThres=Alpha_pos(WeightNumber+1);
    % Initialize recognizing results
    TrainResults=zeros(1,TotalTrainWinNum);
    ValidResults=zeros(1,TotalValidWinNum);
    % Initialize loss
    ValidLoss=0;
    % Get recognizing results of training set
    for WinIndex=1:TotalTrainWinNum
        RealIndex=TrainWinIndex(WinIndex);
        ex=fix((RealIndex-1)/(AcN*WNPA))+1;
        ac=fix((RealIndex-(ex-1)*AcN*WNPA-1)/WNPA)+1;
        win=RealIndex-(ex-1)*AcN*WNPA-(ac-1)*WNPA;
        EInput=reshape([Spike{ex,ac,win,:}],SimNum,EN)';
        [~,TrainResults(WinIndex)]=Iterate_mex(SimNum,EInput,HN,ON,dVGain,...
            WeightEH,WeightHO,SpikeThres,DropMask,DropScale,RestPotential,ac,0);
    end
    % Get accuracy and confusion matrix
    TrainCorrectArray=TrainResults==ExpectTrainResults; 
    for ac=1:AcN
        TrainAcc(e,ac)=sum(TrainCorrectArray(ExpectTrainResults==ac))/(TotalTrainWinNum/AcN);
    end
    for Realac=1:AcN
        for Predictac=1:AcN
            TrainCM(e,Realac,Predictac)=size(find(TrainResults(ExpectTrainResults==Realac)==Predictac),2)/(TotalTrainWinNum/AcN);
        end
    end
    % Probability of hidden layer generate no spike: Training set
    TrainZeroRatio(e)=size(find(TrainResults==0),2)/TotalTrainWinNum;
    % Calculate the loss of validation set, get the results of validation set
    for WinIndex=1:TotalValidWinNum
        RealIndex=ValidWinIndex(WinIndex);
        ex=fix((RealIndex-1)/(AcN*WNPA))+1;
        ac=fix((RealIndex-(ex-1)*AcN*WNPA-1)/WNPA)+1;
        win=RealIndex-(ex-1)*AcN*WNPA-(ac-1)*WNPA;
        EInput=reshape([Spike{ex,ac,win,:}],SimNum,EN)';
        [ValidLossTmp,ValidResults(WinIndex)]=Iterate_mex(SimNum,EInput,HN,ON,dVGain,...
            WeightEH,WeightHO,SpikeThres,DropMask,DropScale,RestPotential,ac,0);
        ValidLoss=ValidLoss-ValidLossTmp;
    end
    ValidScore(e)=ValidLoss/TotalValidWinNum;
    % Get accuracy of validation set
    ValidCorrectArray=ValidResults==ExpectValidResults; 
    for ac=1:AcN
        ValidAcc(e,ac)=sum(ValidCorrectArray(ExpectValidResults==ac))/(TotalValidWinNum/AcN);
    end
    for Realac=1:AcN
        for Predictac=1:AcN
            ValidCM(e,Realac,Predictac)=size(find(ValidResults(ExpectValidResults==Realac)==Predictac),2)/(TotalValidWinNum/AcN);
        end
    end
    % Probability of hidden layer generate no spike: validation set
    ValidZeroRatio(e)=size(find(ValidResults==0),2)/TotalValidWinNum;
    % Record best accuracy
    if mean(ValidAcc(e,:))>mean(MaxAcc(:,2))
        MaxAcc(:,1)=TrainAcc(e,:);
        MaxAcc(:,2)=ValidAcc(e,:);
        MaxAccEpoch=e;
        SNN{1}=reshape(Alpha_pos(1:EHWN),EN,HN);
        SNN{2}=reshape(Alpha_pos(EHWN+1:EHWN+HOWN),HN,ON);
        SNN{3}=[Alpha_pos(WeightNumber+1)];
    end
    TrainScore(e)=Alpha_score;
    SingleTime=toc;
    TotalTime=SingleTime+TotalTime;
    LeftTime=round((TotalTime/e)*(Epoch-e));
    disp(['Sub' num2str(ni) ' Fold' num2str(fold) ' time left: ' char(duration(0,0,LeftTime,'Format','hh:mm:ss')) '. Best Acc: ' num2str(mean(MaxAcc(:,1))) '\' num2str(mean(MaxAcc(:,2)))])
    if Alpha_score<(0.05)
        disp('Train stoped!')
        break;
    end
end
% Get accuracy of testing set
WeightEH=SNN{1};
WeightHO=SNN{2};
SpikeThres=SNN{3}(1);
TestResults=zeros(1,TotalTestWinNum);
for WinIndex=1:TotalTestWinNum
    RealIndex=TestWinIndex(WinIndex);
    ex=fix((RealIndex-1)/(AcN*WNPA))+1;
    ac=fix((RealIndex-(ex-1)*AcN*WNPA-1)/WNPA)+1;
    win=RealIndex-(ex-1)*AcN*WNPA-(ac-1)*WNPA;
    EInput=reshape([Spike{ex,ac,win,:}],SimNum,EN)';
    [~,TestResults(WinIndex)]=Iterate_mex(SimNum,EInput,HN,ON,dVGain,...
        WeightEH,WeightHO,SpikeThres,DropMask,DropScale,RestPotential,ac,0);
end
TestCorrectArray=TestResults==ExpectTestResults; 
for ac=1:AcN
    TestAcc(ac)=sum(TestCorrectArray(ExpectTestResults==ac))/(TotalTestWinNum/AcN);
end
for Realac=1:AcN
    for Predictac=1:AcN
        TestCM(Realac,Predictac)=size(find(TestResults(ExpectTestResults==Realac)==Predictac),2)/(TotalTestWinNum/AcN);
    end
end
Results{1}=TrainResults;
Results{2}=ValidResults;
Results{3}=TestResults;
end