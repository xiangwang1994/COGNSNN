clear;close all;dbstop if error
Epoch=1000;
AcN=5;
SubNum=11;
ExN=10;
load('Results.mat')
% All Loss Function
ALF=zeros(SubNum*ExN,Epoch);
% Train All Accracy
TAA=zeros(SubNum*ExN,Epoch);
% Valid All Accracy
VAA=zeros(SubNum*ExN,Epoch);
% Train No Spike Probability
TNSP=zeros(SubNum*ExN,Epoch);
% Valid No Spike Probability
VNSP=zeros(SubNum*ExN,Epoch);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ni=1:SubNum
    for ex=1:ExN
        TNSP((ni-1)*ExN+ex,:)=TrainZeroRatio{ni,ex};
    end
end
TN=mean(TNSP);
TNSTD=std(TNSP);
figure('NumberTitle', 'off', 'Name', 'The mean probability that all neurons in the hidden layer do not generate spikes');hold on;box on;grid on;
axis([0 Epoch 0 0.01])
yticks(0:0.001:0.01)
yticklabels({'0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'})
plot(TN,'m','LineWidth',1);
set(gca,'FontSize',10,'Fontname','Times new roman');
xlabel('Epoch')
ylabel('Probability (%)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ni=1:SubNum
    for ex=1:ExN
        tmp=TrainScore{ni,ex};
        if ~isempty(find(tmp<0.02)>0)
            tmp(find(tmp<0.02,1):end)=tmp(find(tmp<0.02,1)-1);
        end
        ALF((ni-1)*ExN+ex,:)=tmp;
    end
end
LF=mean(ALF);
LFSTD=std(ALF); 
figure('NumberTitle', 'off', 'Name', 'Loss');hold on;box on;grid on;
axis([0 Epoch 0 2])
yticks(0:0.2:2)
plot(LF-LFSTD,'Color',[255/255 100/255 100/255]);
plot(LF+LFSTD,'Color',[255/255 100/255 100/255]);
plot(LF,'r','LineWidth',1);
P=patch('Faces',1:Epoch*2,'Vertices',[1:Epoch,Epoch:-1:1;LF-LFSTD,fliplr(LF+LFSTD)]','FaceColor','red','FaceAlpha',.3);
P.EdgeColor = 'none';
set(gca,'FontSize',10,'Fontname','Times new roman');
xlabel('Epoch')
ylabel('Loss')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ni=1:SubNum
    for ex=1:ExN
        tmp=mean(ValidAcc{ni,ex},2);
        if ~isempty(find(tmp(2:end)<0.02)>0)
            tmp(find(tmp(2:end)<0.02,1)+1:end)=tmp(find(tmp(2:end)<0.02,1)-1);
        end
        VAA((ni-1)*ExN+ex,:)=tmp;
    end
end
VA=mean(VAA);
VASTD=std(VAA);
figure('NumberTitle', 'off', 'Name', 'The accuracy changing tendency with the epoch.');hold on;box on;grid on;
axis([0 Epoch 0 1])
yticks(0:0.1:1)
yticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
plot(VA-VASTD,'Color',[180/255 180/255 255/255]);
plot(VA+VASTD,'Color',[180/255 180/255 255/255]);
plot(VA,'b','LineWidth',1);
P=patch('Faces',1:Epoch*2,'Vertices',[1:Epoch,Epoch:-1:1;VA-VASTD,fliplr(VA+VASTD)]','FaceColor','blue','FaceAlpha',.3);
P.EdgeColor = 'none';
set(gca,'FontSize',10,'Fontname','Times new roman');
yticklabels({'0','10','20','30','40','50','60','70','80','90','100'})
set(gcf,'position',[0,0,450,300])
xlabel('Epoch')
ylabel('Accuracy (%)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ConMat=zeros(AcN,AcN);
for ni=1:SubNum
    for ex=1:ExN
        ConMat=ConMat+TestCM{ni,ex};
    end
end
ConMat=ConMat./(SubNum*ExN);
figure('NumberTitle', 'off', 'Name', 'Confusion Matrix');axis equal;axis off;hold on
ConMat=flipud(ConMat);
imagesc(ConMat);
colorbar
caxis([0,1])
colorbar('Ticks',0:0.1:1,'TickLabels',{'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'})
for i=1:AcN
    for j=1:AcN
        if (ConMat(i,j)<0.4)
            font_color=[1,1,1];
        else
            font_color=[0,0,0];
        end
        str={[num2str(ConMat(i,j)*100,'%.2f') '%'], num2str(6490*ConMat(i,j))};
        text(j,i,str,'FontName','Times new roman','FontSize',10,'Color',font_color,'HorizontalAlignment','center');
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'The distribution of max accuracy epoch.');hold on;box on;grid on;
n=zeros(1,10);
for i=1:10
    n(i)=length(find(MaxAccEpoch>=(i-1)*100&MaxAccEpoch<i*100));
end
bar(n,0.5)
xlim([0.5 10.5])
ylim([0 20])
xticklabels({'[0,100)','[100,200)','[200,300)','[300,400)','[400,500)','[500,600)','[600,700)','[700,800)','[800,900)','[900,1000)'})
set(gca,'FontSize',10,'Fontname','Times new roman');
set(gcf,'position',[0,0,850,160])
xlabel('Epoch intervals')
ylabel('Frequency')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The accuracy.
Acc=zeros(SubNum,ExN);
for ni=1:SubNum
    for ex=1:ExN
        Acc(ni,ex)=mean(TestAcc{ni,ex});
    end
end
Acc(:,11)=mean(Acc(:,1:10),2);