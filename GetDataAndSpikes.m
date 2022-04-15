clear;close all;dbstop if error
%% 参数
% Subjects
NameList=["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"];
% Subjects number
SubNum=length(NameList);
% Windows length, ms
WL=100;
% Step length
St=50;
% RMSLength
RMSL=150;
% Channel number
ChN=3;
% Feature dimension
FeD=2; 
% Frequency
Frq=1500/1000;
% Gesture number
AcN=5;
% Gesture length
AcL=3000;
% windows number per action
WNPA=(AcL-WL)/St+1;%59
% Experiment number
ExN=10;
Data=cell(1,SubNum);
Spike=cell(1,SubNum);
% Index of training, validating and testing set.
Index=cell(ExN,3);
ValidRatio=2/9;
ValidWinNum=ceil(ValidRatio*WNPA);
TrainWinNum=WNPA-ValidWinNum;
% 8-order band pass filters:F1,15-100;F2,100-550;F3,15-550. 2-order notch. filter:F4,50Hz
load Filters.mat
%% Generate SFD spikes
for ni=1:SubNum
    testname=NameList(ni);
    Route=char(strcat(".\sEMG\",testname,"\"));
    Data{ni}=SFDLoadEMG(Route,ExN,ChN,AcN,AcL,Frq,F,FeD,RMSL);
end
for ni=1:SubNum
    Spike{ni}=SFDExtractSpike(Data{ni},ExN,AcN,AcL,Frq,WL,St,ChN,FeD);
end
save('Spikes.mat','Spike')
%% Generate index
for ex=1:ExN
    [~,IndexTmp]=sort(rand(WNPA,AcN*(ExN-1)));
    TestWinIndex=(ex-1)*WNPA*AcN+1:ex*WNPA*AcN;
    TrainWinIndex=find(IndexTmp<=TrainWinNum);
    TrainWinIndex(TrainWinIndex>(ex-1)*WNPA*AcN)=TrainWinIndex(TrainWinIndex>(ex-1)*WNPA*AcN)+WNPA*AcN;
    ValidWinIndex=find(IndexTmp>TrainWinNum&IndexTmp<=TrainWinNum+ValidWinNum);
    ValidWinIndex(ValidWinIndex>(ex-1)*WNPA*AcN)=ValidWinIndex(ValidWinIndex>(ex-1)*WNPA*AcN)+WNPA*AcN;
    Index{ex,1}=TrainWinIndex;
    Index{ex,2}=ValidWinIndex;
    Index{ex,3}=TestWinIndex';
end
save('Index.mat','Index')
%% SFDLoadEMG
function SFDData=SFDLoadEMG(Route,ExN,ChN,AcN,AcL,Frq,F,FeD,RMSL)
RMSL=RMSL*Frq;
AcL=AcL*Frq;
OriginalData=zeros(ExN,ChN,AcL*AcN);
Data=zeros(ExN,ChN,AcL*AcN);
FilterData=zeros(ExN,ChN*FeD,AcL*AcN);
RMSData=zeros(ExN,ChN,AcL*AcN);
RMSFilterData=zeros(ExN,ChN*FeD,AcL*AcN);
SFDData=zeros(ExN,ChN*(FeD+1),AcL*AcN);
acup=7500*Frq+AcL/2;
acdn=7500*Frq-AcL/2;
for ex=1:ExN
    DataBuffer=load([Route num2str(ex) '.mat']);
    OriginalDataTmp=zeros(ChN,length(DataBuffer.Data{1}));
    NotchDataTmp=zeros(ChN,length(DataBuffer.Data{1}));
    DataTmp=zeros(ChN,length(DataBuffer.Data{1}));
    FilterDataTmp=zeros(ChN*FeD,length(DataBuffer.Data{1}));
    RMSDataTmp=zeros(ChN,length(DataBuffer.Data{1}));
    RMSFilterDataTmp=zeros(ChN*FeD,length(DataBuffer.Data{1}));
    TimeIndex=0:size(DataBuffer.Data{1},1)-1;
    for ch=1:ChN
        OriginalDataTmp(ch,:)=(DataBuffer.Data{ch}');
        NotchDataTmp(ch,:)=filter(F{4},OriginalDataTmp(ch,:));
        DataTmp(ch,:)=filter(F{3},NotchDataTmp(ch,:));
        for FilterIndex=1:FeD
            FilterDataTmp((ch-1)*FeD+FilterIndex,:)=filter(F{FilterIndex},NotchDataTmp(ch,:));
        end
    end
    iOffset=RMSL-1;
    iStop=size(DataBuffer.Data{1},1)-RMSL+1;
    for ch=1:ChN
        for i=1:iStop
            RMSDataTmp(ch,i)=rms(DataTmp(ch,i:i+iOffset));
        end
        for fed=1:FeD
            for i=1:iStop
                RMSFilterDataTmp((ch-1)*FeD+fed,i)=rms(FilterDataTmp((ch-1)*FeD+fed,i:i+iOffset));
            end
        end
    end
    for ac=1:AcN
        in=intersect(find(TimeIndex>=acdn+(ac-1)*15000),find(TimeIndex<acup+(ac-1)*15000));
        OriginalData(ex,:,((ac-1)*AcL+1):(ac*AcL))=OriginalDataTmp(:,in);
        Data(ex,:,((ac-1)*AcL+1):(ac*AcL))=abs(DataTmp(:,in));
        FilterData(ex,:,((ac-1)*AcL+1):(ac*AcL))=abs(FilterDataTmp(:,in));
        RMSData(ex,:,((ac-1)*AcL+1):(ac*AcL))=abs(RMSDataTmp(:,in-iOffset));
        RMSFilterData(ex,:,((ac-1)*AcL+1):(ac*AcL))=abs(RMSFilterDataTmp(:,in-iOffset));
    end
end
% Normalize
for ex=1:ExN
    for ch=1:ChN
        RMSDataSort=sort(RMSData(ex,ch,:));
        RMSDataMax=RMSDataSort(fix((AcL*AcN)*0.95));
        RMSData(ex,ch,:)=RMSData(ex,ch,:)/RMSDataMax;
        for fe=1:FeD
            RMSFilterDataSort=sort(RMSFilterData(ex,(ch-1)*FeD+fe,:));
            RMSFilterDataMax=RMSFilterDataSort(fix((AcL*AcN)*0.95));
            RMSFilterData(ex,(ch-1)*FeD+fe,:)=RMSFilterData(ex,(ch-1)*FeD+fe,:)/RMSFilterDataMax;
        end
    end
end
RMSData(RMSData>1)=1;
RMSFilterData(RMSFilterData>1)=1;
for ex=1:ExN
    for ch=1:ChN
        SFDData(ex,(ch-1)*(FeD+1)+1,:)=RMSData(ex,ch,:);
        SFDData(ex,(ch-1)*(FeD+1)+2,:)=RMSFilterData(ex,(ch-1)*FeD+1,:);
        SFDData(ex,(ch-1)*(FeD+1)+3,:)=RMSFilterData(ex,(ch-1)*FeD+2,:);
    end
end
end
%% SFDExtractSpike
function SFDSpike=SFDExtractSpike(SFDData,ExN,AcN,AcL,Frq,WL,St,ChN,FeD)
AcL=AcL*Frq;
WL=WL*Frq;
St=St*Frq;
RMSData=zeros(ExN,ChN,AcL*AcN);
RMSFilterData=zeros(ExN,ChN*FeD,AcL*AcN);
for ex=1:ExN
    for ch=1:ChN
        RMSData(ex,ch,:)=SFDData(ex,(ch-1)*(FeD+1)+1,:);
        RMSFilterData(ex,(ch-1)*FeD+1,:)=SFDData(ex,(ch-1)*(FeD+1)+2,:);
        RMSFilterData(ex,(ch-1)*FeD+2,:)=SFDData(ex,(ch-1)*(FeD+1)+3,:);
    end
end
% Windows number per gesture
WNPA=(AcL-WL)/St+1;%49
RMSSpike=cell(ExN,AcN,WNPA,ChN*FeD);
RMSFilterSpike=cell(ExN,AcN,WNPA,ChN*FeD);
SFDSpike=cell(ExN,AcN,WNPA,ChN*(FeD+1));
SpikeThres=2;
for ex=1:ExN
    for ac=1:AcN
        for win=1:WNPA
            Tmp=squeeze(RMSData(ex,:,(ac-1)*AcL+(win-1)*St+1:(ac-1)*AcL+(win-1)*St+WL));
            for i=1:size(Tmp,1)
                SpikeTmp=zeros(1,length(Tmp));
                SpikeValue=0;
                for j=1:length(Tmp)
                    SpikeValue=SpikeValue+Tmp(i,j);
                    % if the sum exceed the threshold, then generate a spike
                    if SpikeValue>=SpikeThres
                        SpikeValue=0;
                        SpikeTmp(j)=1;
                    end
                end
                RMSSpike{ex,ac,win,i}=SpikeTmp;
            end
            Tmp=squeeze(RMSFilterData(ex,:,(ac-1)*AcL+(win-1)*St+1:(ac-1)*AcL+(win-1)*St+WL));
            for i=1:size(Tmp,1)
                SpikeTmp=zeros(1,length(Tmp));
                SpikeValue=0;
                for j=1:length(Tmp)
                    SpikeValue=SpikeValue+Tmp(i,j);
                    if SpikeValue>=SpikeThres
                        SpikeValue=0;
                        SpikeTmp(j)=1;
                    end
                end
                RMSFilterSpike{ex,ac,win,i}=SpikeTmp;
            end
        end
    end
end
for ex=1:ExN
    for ac=1:AcN
        for win=1:WNPA
            SFDSpike{ex,ac,win,1}=RMSSpike{ex,ac,win,1};
            SFDSpike{ex,ac,win,2}=RMSFilterSpike{ex,ac,win,1};
            SFDSpike{ex,ac,win,3}=RMSFilterSpike{ex,ac,win,2};
            SFDSpike{ex,ac,win,4}=RMSSpike{ex,ac,win,2};
            SFDSpike{ex,ac,win,5}=RMSFilterSpike{ex,ac,win,3};
            SFDSpike{ex,ac,win,6}=RMSFilterSpike{ex,ac,win,4};
            SFDSpike{ex,ac,win,7}=RMSSpike{ex,ac,win,3};
            SFDSpike{ex,ac,win,8}=RMSFilterSpike{ex,ac,win,5};
            SFDSpike{ex,ac,win,9}=RMSFilterSpike{ex,ac,win,6};
        end
    end
end
end