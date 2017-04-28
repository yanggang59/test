%hog_libsvm.m
clc; 
clear ;  
%%training 
ReadList1  = textread('pos_list.txt','%s','delimiter','\n');%loading positive samples
sz1=size(ReadList1);    
label1=ones(sz1(1),1); %positive sample labels

ReadList2  = textread('neg_list.txt','%s','delimiter','\n');%loading negtive samples
sz2=size(ReadList2);  
label2=zeros(sz2(1),1);%negtive sample labels

label=[label1',label2']';%labels put together
total_num=length(label);  %length of the label
data=zeros(total_num,1764);  %initializing the hog discriptor,1764D

%Read positive features,and calculate hog descriptors
for i=1:sz1(1)  
   name= char(ReadList1(i,1));  
   image=imread(strcat('D:\MATLAB7\work\svm_test\pos\',name)); 
   im=imresize(image,[64,64]);  
   img=rgb2gray(im);  
   hog =hogcalculator(img);  
   data(i,:)=hog;  
end  

%Read negative features,calculate hog descriptors
for j=1:sz2(1)  
   name= char(ReadList2(j,1));  
   image=imread(strcat('D:\MATLAB7\work\svm_test\neg\',name));  
   im=imresize(image,[64,64]);  
   img=rgb2gray(im);  
   hog =hogcalculator(img);  
   data(sz1(1)+j,:)=hog;  
end  

model=svmtrain(label,data,'-t 0 -g 2.8');%training using rbf kernel ,r=2.8,-t 2 is (default) rbf,-t 0 is linear 
%model=svmtrain(label,data)
save model model;
[predict_label,accuracy,unknow] = svmpredict(label,data,model,'q');  %using training set to test
%load model; 

%generating testing set
ReadList_test = textread('test_pic.txt','%s','delimiter','\n');
sz_test=size(ReadList_test);
label_test=[1;1;1;0;0;0];
num=length(label_test);
data_test=zeros(num,1764);
for k=1:sz_test(1)
   name= char(ReadList_test(k,1));  
   image=imread(strcat('D:\MATLAB7\work\svm_test\test\test',name)); 
   im=imresize(image,[64,64]);  
   img=rgb2gray(im);  
   hog =hogcalculator(img);  
   data_test(k,:)=hog;  
end  

[predict_label_test,accuracy_test,unknow_test]= svmpredict(label_test,data_test,model,'q');




