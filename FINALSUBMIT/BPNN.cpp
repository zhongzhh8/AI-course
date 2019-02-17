#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <cstdio>
#include <set>
#include <vector>
#include <ctime>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <fstream>
using namespace std;
bool savew=false;
int Length;//w、train向量的长度 
int hidenum=70;
int iter_time;
bool Random_init=true;
double eta;
int traincnt;//样本数 
int testcnt;
int valicnt;
double train[20000][100];//可容纳5000个train样本 
double vali[10000][100];
double test[2500][100];
int trainlabel[20000];//每个样本的标签（正负） 
int valilabel[10000];
double trainpredict[20000];
double testpredict[1000];
double valipredict[10000];
double MT[10000];//训练集MSE下降过程 
double MV[10000];//验证集MSE下降过程  
double x[100];//每次取一个样本，x数组存储其所有特征，即输入层节点数组 
double h[100];//隐藏层节点数组
double y;//只有一个输出层节点
double MSE;//训练集均方误差 
double MSE_of_vali;//验证集均方误差 
double T_W_i2h[100][100];//输入层到隐藏层
double T_W_h2o[100];//隐藏层到输出层
double W_i2h[100][100];//输入层到隐藏层的权重数组 
double W_h2o[100];//隐藏层到输出层的权重数组
double delta_out;//输出层误差梯度？δ
double delta_hide[100];//隐藏层误差梯度？δj 
int test_date[1000];
void SetParameter(){
	eta=0.01;
	iter_time=1000;
	Random_init=true;
	savew=true;
}
int WhatDay(string str)
{
	string year="",month="",date="";
	int which=0;
	for(int j=0;j<str.size();j++){
		if(str[j]=='/') which++;
		if(which==0&&str[j]!='/')	year=year+str[j];
		else if(which==1&&str[j]!='/')	month+=str[j];
		else if(which==2&&str[j]!='/')	date+=str[j];
	}
	int yearnum,monthnum,datenum;
	stringstream sss;
	sss.clear();
	sss.str(year);
	sss>>yearnum;	
	sss.clear();
	sss.str(month);
	sss>>monthnum;
	sss.clear();
	sss.str(date);
	sss>>datenum;
	int day_code,s=0,a[12]={31,28,31,30,31,30,31,31,30,31,30,31};
	day_code=(yearnum+(yearnum-1)/4-(yearnum-1)/100+(yearnum-1)/400)%7;
	for(int i=0;i<monthnum-1;i++)
		s=s+a[i];
	s=s+datenum;
	if(yearnum%4==0) s=s+1;
	int j=(s+day_code-1)%7;
	return j;
}
int WhatDate(string str)
{
	string year="",month="",date="";
	int which=0;
	for(int j=0;j<str.size();j++){
		if(str[j]=='/') which++;
		if(which==0&&str[j]!='/')	year=year+str[j];
		else if(which==1&&str[j]!='/')	month+=str[j];
		else if(which==2&&str[j]!='/')	date+=str[j];
	}
	int datenum;
	stringstream sss;
	sss.clear();
	sss.str(date);
	sss>>datenum;
	return datenum;
}
int WhatMonth(string str){
	string year="",month="",date="";
	int which=0;
	for(int j=0;j<str.size();j++){
		if(str[j]=='/') which++;
		if(which==0&&str[j]!='/')	year=year+str[j];
		else if(which==1&&str[j]!='/')	month+=str[j];
		else if(which==2&&str[j]!='/')	date+=str[j];
	}
	int yearnum,monthnum,datenum;
	stringstream sss;
	sss.clear();
	sss.str(month);
	sss>>monthnum;
	return monthnum;
}
int WhatYear(string str){
	string year="",month="",date="";
	int which=0;
	for(int j=0;j<str.size();j++){
		if(str[j]=='/') which++;
		if(which==0&&str[j]!='/')	year=year+str[j];
		else if(which==1&&str[j]!='/')	month+=str[j];
		else if(which==2&&str[j]!='/')	date+=str[j];
	}
	int yearnum,monthnum,datenum;
	stringstream sss;
	sss.clear();
	sss.str(year);
	sss>>yearnum;
	return yearnum;
}
void Readtrain()
{
	ifstream fin("train.csv");
	string line;
	vector<string> fields;
	traincnt=0;
	valicnt=0;
	int mo=8;
	int cnt=0;
	while(getline(fin,line))
	{
		cnt++;
		if(cnt==1) continue;
		fields.clear();
		istringstream sin(line);		
		string field;
		while(getline(sin,field,','))
		{
			fields.push_back(field);
		}	
		train[traincnt][0]=1;
		vali[valicnt][0]=1;
		int index=1;
		for(int i=0;i<fields.size();i++)
		{
			if(i==0) continue;
			double temnum;
			stringstream ss;
			ss.clear();
			ss.str(fields[i]);
			ss>>temnum;			
			if(cnt%mo==0)//vali 
			{
				if(i==1){
					int month=WhatMonth(fields[i]);
					int date=WhatDate(fields[i]);
					if(month==12&&(date>=22&&date<=30)) vali[valicnt][index++]=1;
					else vali[valicnt][index++]=0;
					
					int year=WhatYear(fields[i]);
					if(year==2011) vali[valicnt][index++]=0;
					else vali[valicnt][index++]=1;
					
					int day=WhatDay(fields[i]);
					for(int k=0;k<7;k++){
						if(k==day) vali[valicnt][index]=1;
						else vali[valicnt][index]=0;
						if(k!=6) index++;
					}
				}
				else if(i==2){			
					for(int k=0;k<24;k++){
						if(k==temnum) vali[valicnt][index]=1;
						else vali[valicnt][index]=0;
						if(k!=23) index++;
					}
				}
				else if(i==3){
					if(temnum==1){
						vali[valicnt][index]=1;vali[valicnt][index+1]=0;vali[valicnt][index+2]=0;
					}
					else if(temnum==2) {
						vali[valicnt][index]=0;vali[valicnt][index+1]=1;vali[valicnt][index+2]=0;
					}
					else{
						vali[valicnt][index]=0;vali[valicnt][index+1]=0;vali[valicnt][index+2]=1;
					} 
					index+=2;
				}
				else if(i==fields.size()-1) {
					if(temnum<2000){
						valilabel[valicnt]=temnum;
						valicnt++;
					}	
				}
				else vali[valicnt][index]=temnum;			
			}
			else//train 
			{				
				if(i==1){
					int month=WhatMonth(fields[i]);
					int date=WhatDate(fields[i]);
					if(month==12&&(date>=22&&date<=30)) train[traincnt][index++]=1;
					else train[traincnt][index++]=0;
					
					int year=WhatYear(fields[i]);
					if(year==2011) train[traincnt][index++]=0;
					else train[traincnt][index++]=1;
					
					int day=WhatDay(fields[i]);
					for(int k=0;k<7;k++){
						if(k==day) train[traincnt][index]=1;
						else train[traincnt][index]=0;
						if(k!=6) index++;
					}
				}
				else if(i==2){			
					for(int k=0;k<24;k++){
						if(k==temnum) train[traincnt][index]=1;
						else train[traincnt][index]=0;
						if(k!=23) index++;
					}
				}
				else if(i==3){
					if(temnum==1){
						train[traincnt][index]=1;train[traincnt][index+1]=0;train[traincnt][index+2]=0;
					}
					else if(temnum==2) {
						train[traincnt][index]=0;train[traincnt][index+1]=1;train[traincnt][index+2]=0;
					}
					else{
						train[traincnt][index]=0;train[traincnt][index+1]=0;train[traincnt][index+2]=1;
					} 
					index+=2;
				}
				else if(i==fields.size()-1) 
				{
					if(temnum<2000){
						trainlabel[traincnt]=temnum;
						traincnt++;
					}
				}
				else train[traincnt][index]=temnum;				
			}
			index++;
		}						
		Length=index-1;
	}
}

void Readtest()
{
	ifstream fin("test.csv");
	string line;
	vector<string> fields;
	testcnt=0;
	int cnt=0;
	while(getline(fin,line))
	{
		cnt++;
		if(cnt==1) continue;
		fields.clear();
		istringstream sin(line);		
		string field;
		while(getline(sin,field,',')){
			fields.push_back(field);
		}	
		test[testcnt][0]=1;
		int index=1;
		for(int i=0;i<fields.size()-1;i++)
		{
			if(i==0) continue;
			double temnum;
			stringstream ss;
			ss.clear();
			ss.str(fields[i]);
			ss>>temnum;	
			if(i==1){
				int month=WhatMonth(fields[i]);
				int date=WhatDate(fields[i]);
				if(month==12&&(date>=22&&date<=30)) test[testcnt][index++]=1;
				else test[testcnt][index++]=0;
				
				int year=WhatYear(fields[i]);
				if(year==2011) test[testcnt][index++]=0;
				else test[testcnt][index++]=1;
				
				
				int day=WhatDay(fields[i]);
				for(int k=0;k<7;k++){
					if(k==day) test[testcnt][index]=1;
					else test[testcnt][index]=0;
					if(k!=6) index++;
				}
				test_date[testcnt]=date;
			}
			else if(i==2){			
				for(int k=0;k<24;k++){
					if(k==temnum) test[testcnt][index]=1;
					else test[testcnt][index]=0;
					if(k!=23) index++;
				}
			}
			else if(i==3){
				if(temnum==1){
					test[testcnt][index]=1;test[testcnt][index+1]=0;test[testcnt][index+2]=0;
				}
				else if(temnum==2) {
					test[testcnt][index]=0;test[testcnt][index+1]=1;test[testcnt][index+2]=0;
				}
				else{
					test[testcnt][index]=0;test[testcnt][index+1]=0;test[testcnt][index+2]=1;
				} 
				index+=2;
			}
			else test[testcnt][index]=temnum;
			index++;
		}
		testcnt++;
	}
}
void readpreviousW() 
{
	ifstream fin("Winit.csv");
	for(int i=0;i<Length;i++)
		for(int j=0;j<hidenum;j++)
			fin>>W_i2h[i][j];
	for(int i=0;i<hidenum;i++) fin>>W_h2o[i];
}
void initialize_weight()
{
	if(Random_init==true){
		srand(time(0));	
		for(int i=0;i<Length;i++)//初始化输入层到隐藏层之间权值向量W_i2h
			for(int j=0;j<hidenum;j++)
				W_i2h[i][j]=rand()*1.0/RAND_MAX*2;	
		for(int i=0;i<hidenum;i++) W_h2o[i]=rand()*1.0/RAND_MAX*2;//初始化隐藏层到输出层之间权值向量W_h2o
	}
	else readpreviousW();
}
void initialize_T_W()
{	
	for(int i=0;i<Length;i++)//初始化T_W_i2h，即△Wij 
		for(int j=0;j<hidenum;j++)
			T_W_i2h[i][j]=0;
	for(int i=0;i<hidenum;i++) T_W_h2o[i]=0;//初始化T_W_h2o，即△Wi
}
void forward_pass_i2h()//输入层到隐藏层 
{
	h[0]=1;//为阈值所准备 
	for(int i=1;i<hidenum;i++)
	{
		double in=0;//首先计算输入到隐藏层节点的值 
		for(int j=0;j<Length;j++) in+=W_i2h[j][i]*x[j];
		h[i]=1/(1+exp(-1*in));//以sigmoid函数为激活函数，确定隐藏层节点数值 
		//h[i]=( exp(in)-exp(-in) )/( exp(in)+exp(-in) );//tanh
	}
}
void forward_pass_h2o(int index)//隐藏层到输出层 
{
	double in=0;//首先计算输入到输出层节点的值 
	for(int i=0;i<hidenum;i++) in+=W_h2o[i]*h[i];
	y=in;//以线性函数f(x)=x为激活函数，确定输出层节点数值 
}
void backward_pass(int index)//反向传递 
{
	delta_out=trainlabel[index]-y;
	for(int i=0;i<hidenum;i++)
		delta_hide[i]=delta_out*W_h2o[i]*h[i]*(1-h[i]);	
		//delta_hide[i]=delta_out*W_h2o[i]*(1+h[i])*(1-h[i]);  //tanh 
	//累加权值向量W_h2o的更新值 
	for(int i=0;i<hidenum;i++) T_W_h2o[i]+=delta_out*h[i];	
	//累加权值向量W_i2h的更新值 
	for(int j=0;j<hidenum;j++) 
		for(int i=0;i<Length;i++)
			T_W_i2h[i][j]+=delta_hide[j]*x[i];
}
void Update_Weight()//更新权值变量 
{
	for(int j=0;j<hidenum;j++) //更新W_i2h 
		for(int i=0;i<Length;i++)
			W_i2h[i][j]+=eta*T_W_i2h[i][j]/traincnt;
	for(int i=0;i<hidenum;i++) W_h2o[i]+=eta*T_W_h2o[i]/traincnt;	//更新W_h2o 
}
void Use_train()
{
	MSE=0;
	for(int k=0;k<traincnt;k++)
	{
		h[0]=1;
		for(int i=1;i<hidenum;i++)
		{
			double sum=0;
			for(int j=0;j<Length;j++) sum+=W_i2h[j][i]*train[k][j];
			h[i]=1/(1+exp(-1*sum));
			//h[i]=( exp(sum)-exp(-sum) )/( exp(sum)+exp(-sum) );
		}
		double sum=0;
		for(int i=0;i<hidenum;i++) {
			sum+=W_h2o[i]*h[i];
		}
		if(sum<0) sum=10;
		trainpredict[k]=sum;
		MSE+=(sum-trainlabel[k])*(sum-trainlabel[k]);
	}
	MSE/=traincnt;
}
void Use_vali()
{
	for(int k=0;k<valicnt;k++)
	{
		h[0]=1;
		for(int i=1;i<hidenum;i++)
		{
			double sum=0;
			for(int j=0;j<Length;j++) sum+=W_i2h[j][i]*vali[k][j];
			h[i]=1/(1+exp(-1*sum));
		}
		double sum=0;
		for(int i=0;i<hidenum;i++) sum+=W_h2o[i]*h[i];
		if(sum<0) sum=10;
		valipredict[k]=sum;
		MSE_of_vali+=(sum-valilabel[k])*(sum-valilabel[k]);
	}
	MSE_of_vali/=valicnt;
}
void Use_test()
{
	int testpcnt=0;
	for(int k=0;k<testcnt;k++)
	{
		h[0]=1;
		for(int i=1;i<hidenum;i++)
		{
			double sum=0;
			for(int j=0;j<Length;j++) sum+=W_i2h[j][i]*test[k][j];
			h[i]=1/(1+exp(-1*sum));
		}
		double sum=0;
		for(int i=0;i<hidenum;i++) sum+=W_h2o[i]*h[i];
		if(sum<0) sum=5;
		if(test_date[k]>=22) sum=sum*0.6;
		testpredict[testpcnt++]=sum;
	}
}
void Output_testpredict()
{
	ofstream fout;
	fout.open("15352446_zhongzhanhui.csv");
	for(int i=0;i<testcnt;i++)
		fout<<(int)testpredict[i]<<endl;
	fout.close();
}
void OutputMSE()
{
	ofstream fout;
	fout.open("MSE.csv");
	for(int i=0;i<50;i++)	fout<<MT[i]<<',';
	fout<<endl;
	for(int i=0;i<50;i++)	fout<<MV[i]<<',';
	fout.close();
}
void Output_compare()
{
	ofstream fout;
	fout.open("traincompare.csv");
	int start=0,end=traincnt;
	start=1000;end=1700;
	for(int i=start;i<end;i++)		
		fout<<trainlabel[i]<<',';
	fout<<endl;	
	for(int i=start;i<end;i++)
		fout<<trainpredict[i]<<',';
	fout<<endl;	
	fout.close();
	fout.open("valicompare.csv");
	start=0;end=valicnt;
	//start=1000;end=1300;
	for(int i=start;i<end;i++)		
		fout<<valilabel[i]<<',';
	fout<<endl;	
	for(int i=start;i<end;i++)
		fout<<valipredict[i]<<',';
	fout<<endl;			
	fout.close();
}
void nomalization(){
	for(int i=0;i<Length;i++){
		double maxnum=0,minnum=10000;
		for(int j=0;j<traincnt;j++){
			if(train[j][i]>maxnum) maxnum=train[j][i];
			if(train[j][i]<minnum) minnum=train[j][i];
		}
		for(int j=0;j<traincnt;j++){
			if(maxnum!=minnum) train[j][i]=(train[j][i]-minnum)/(maxnum-minnum);
		}
	}
	for(int i=0;i<Length;i++){
		double maxnum=0,minnum=10000;
		for(int j=0;j<valicnt;j++){
			if(vali[j][i]>maxnum) maxnum=vali[j][i];
			if(vali[j][i]<minnum) minnum=vali[j][i];
		}
		for(int j=0;j<valicnt;j++){
			if(maxnum!=minnum) vali[j][i]=(vali[j][i]-minnum)/(maxnum-minnum);
		}
	}	
	for(int i=0;i<Length;i++){
		double maxnum=0,minnum=10000;
		for(int j=0;j<testcnt;j++){
			if(test[j][i]>maxnum) maxnum=test[j][i];
			if(test[j][i]<minnum) minnum=test[j][i];
		}
		for(int j=0;j<testcnt;j++){
			if(maxnum!=minnum)	test[j][i]=(test[j][i]-minnum)/(maxnum-minnum);
		}
	}
}
void saveW()
{
	ofstream fout;
	fout.open("Winit.csv");
	for(int i=0;i<Length;i++)
		for(int j=0;j<hidenum;j++)
			fout<<W_i2h[i][j]<<endl;
	for(int i=0;i<hidenum;i++) fout<<W_h2o[i]<<endl;
	fout.close();
}

int main()
{
	SetParameter();//设置各种参数如学习步长、迭代次数、选择W随机初始化还是读取上次保存下来的W 
	Readtrain();//读取训练集，且其中训练集每8个样本取前7个作为训练集，第8个作为验证集
	Readtest();//读取测试集 
	nomalization();//特征值归一化 
	cout<<"Length="<<Length<<"traincnt="<<traincnt<<" valicnt="<<valicnt<<" testcnt="<<testcnt<<endl;
	initialize_weight();//初始化权值向量 Wij和 Wi
	for(int k=0;k<iter_time;k++)//迭代iter_time次 
	{
		MSE=0;//训练集的MSE 
		MSE_of_vali=0;//验证集的MSE 
		initialize_T_W();//初始化△Wij和△Wi为全零向量 
		for(int i=0;i<traincnt;i++)//批梯度下降，遍历整个训练集 
		{	
			for(int j=0;j<Length;j++) x[j]=train[i][j];//取出当前遍历到的样本的特征值放进x数组 
			forward_pass_i2h();	//正向传递，从输入层到隐藏层 
			forward_pass_h2o(i);//正向传递，从隐藏层到输出层 
			backward_pass(i);//反向传递，累计△Wij和△Wi
		}
		Update_Weight();//每遍历完一次训练集更新一次权值向量 
		if(k%10==0){//每迭代十次计算并输出训练集MSE查看学习效果 
			Use_train();//用当前的权值向量预测训练集结果并计算MSE 
			Use_vali();//用当前的权值向量预测验证集结果并计算MSE 
			MT[k/10]=MSE;	
			MV[k/10]=MSE_of_vali;
			cout<<"训练集MSE:"<<MSE<<endl;
		}
	}
	OutputMSE();//将MSE数组输出到文件 
	Output_compare();//将训练集最终预测cnt和真实cnt输出到同一个文件中对比 
	Use_test();//使用权值向量预测 测试集 
	Output_testpredict();//将测试集预测结果输出到文件（即要提交的csv文件）	
	if(savew==true) saveW();//把权值变量保存下来 
} 










