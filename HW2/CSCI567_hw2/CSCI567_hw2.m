function [crsnt, dict, email_train_data, email_train_label, ion_train_data, ion_train_label] = CSCI567_hw2(  )
%CSCI567_HW2 Summary of this function goes here
%   Detailed explanation goes here

% add subfolders to path

path = pwd;
addpath(genpath(strcat(path,'/myFiles')));
addpath(genpath(strcat(path,'/hw2_data')));

linestyles = cellstr(char('-',':','-.','--','-',':','-.','--','-',':','-',':',...
'-.','--','-',':','-.','--','-',':','-.'));
Markers=['o','x','+','*','s','d','v','^','<','>','p','h','.',...
'+','*','o','x','^','<','h','.','>','p','s','d','v',...
'o','x','+','*','s','d','v','^','<','>','p','h','.'];


[dict, email_train_data, email_train_label] = top3('train');
[dict, email_test_data, email_test_label] = top3('test'); %but this doesn't calculate top 3 in test data.

ion_train_data = importIonosphereFeatures('ionosphere_train.dat');
ion_train_label = importIonosphereLabel('ionosphere_train.dat');
ion_train_label = double(strcmp(ion_train_label,'b'));

ion_test_data = importIonosphereFeatures('ionosphere_test.dat');
ion_test_label = importIonosphereLabel('ionosphere_test.dat');
ion_test_label = double(strcmp(ion_test_label,'b'));

disp('------------------');
disp('Email Data');
[newton_weight_email_training,newton_bais_email_training] = batchGradientDescent(email_train_data, email_train_label,'Email Data', linestyles, Markers);
disp('------------------');

disp('Email Data Testing');
[newton_weight_email_testing,newton_bais_email_testing] = batchGradientDescent(email_test_data, email_test_label,'Email Data Testing', linestyles, Markers);
disp('------------------');


disp('Ionosphere Data Training');
[newton_weight_iono_training, newton_bias_iono_training] = batchGradientDescent(ion_train_data, ion_train_label, 'Ionosphere Data', linestyles, Markers);
disp('------------------');


disp('Ionosphere Data Testing');
[newton_weight_iono_testing, newton_bias_iono_testing] = batchGradientDescent(ion_test_data, ion_test_label, 'Ionosphere Data', linestyles, Markers);
disp('------------------');


crossEntropyVSRegu(email_train_data, email_train_label,email_test_data,email_test_label,'Email Data');
crossEntropyVSRegu(ion_train_data, ion_train_label,ion_test_data,ion_test_label,'Ionosphere Data');


%Training
newtonsMethod(email_train_data, email_train_label,'Email Data Training',newton_weight_email_training,newton_bais_email_training);

%Testing
newtonsMethod(email_test_data, email_test_label,'Email Data Testing',newton_weight_email_testing,newton_bais_email_testing);



%Training
newtonsMethod(ion_train_data, ion_train_label,'Ionosphere Data',newton_weight_iono_training, newton_bias_iono_training);

%Testing
newtonsMethod(ion_test_data, ion_test_label,'Ionosphere Data',newton_weight_iono_testing, newton_bias_iono_testing);
end


