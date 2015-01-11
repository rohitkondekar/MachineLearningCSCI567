function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CSCI 576 2014 Fall, Homework 1

% Class Values: 
% 
% unacc, acc, good, vgood 


% Attributes: 
% 
% buying: vhigh, high, med, low. 
% maint: vhigh, high, med, low. 
% doors: 2, 3, 4, 5more. 
% persons: 2, 4, more. 
% lug_boot: small, med, big. 
% safety: low, med, high. 

%calculate prior probability

% boolean array 0/1 for strcmp
bool_unacc = strcmp('unacc',train_label);
bool_acc = strcmp('acc',train_label);
bool_good = strcmp('good',train_label);
bool_vgood = strcmp('vgood',train_label);

%counts
c_unacc = sum(bool_unacc);
c_acc = sum(bool_acc);
c_good = sum(bool_good);
c_vgood = sum(bool_vgood);


%Two ways to feed in probability
%manual
%auto

%prior calculated manually including validation data
% 1339
% 
% 308     ---- 0.230022
% 49      ---- 0.036594
% 924     ---- 0.690067
% 58      ---- 0.043315
% 
 prior_acc = 0.230022;
 prior_good = 0.036594;
 prior_unacc = 0.690067;
 prior_vgood = 0.043315;

% To calculate automatically from training set ----- >
% c_total = c_unacc+c_acc+c_good+c_vgood;
% prior_unacc = c_unacc/c_total;
% prior_acc = c_acc/c_total;
% prior_good = c_good/c_total;
% prior_vgood = c_vgood/c_total;


% boolean array for buying feature
bool_buying_vhigh = strcmp('vhigh',train_data(:,1));
bool_buying_high = strcmp('high',train_data(:,1));
bool_buying_med = strcmp('med',train_data(:,1));
bool_buying_low = strcmp('low',train_data(:,1));

% conditional probability for buying
p_buying_vhigh_unacc = sum(and(bool_buying_vhigh,bool_unacc))/c_unacc;
p_buying_vhigh_acc = sum(and(bool_buying_vhigh,bool_acc))/c_acc;
p_buying_vhigh_good = sum(and(bool_buying_vhigh,bool_good))/c_good;
p_buying_vhigh_vgood = sum(and(bool_buying_vhigh,bool_vgood))/c_vgood;

p_buying_high_unacc = sum(and(bool_buying_high,bool_unacc))/c_unacc;
p_buying_high_acc = sum(and(bool_buying_high,bool_acc))/c_acc;
p_buying_high_good = sum(and(bool_buying_high,bool_good))/c_good;
p_buying_high_vgood = sum(and(bool_buying_high,bool_vgood))/c_vgood;

p_buying_med_unacc = sum(and(bool_buying_med,bool_unacc))/c_unacc;
p_buying_med_acc = sum(and(bool_buying_med,bool_acc))/c_acc;
p_buying_med_good = sum(and(bool_buying_med,bool_good))/c_good;
p_buying_med_vgood = sum(and(bool_buying_med,bool_vgood))/c_vgood;

p_buying_low_unacc = sum(and(bool_buying_low,bool_unacc))/c_unacc;
p_buying_low_acc = sum(and(bool_buying_low,bool_acc))/c_acc;
p_buying_low_good = sum(and(bool_buying_low,bool_good))/c_good;
p_buying_low_vgood = sum(and(bool_buying_low,bool_vgood))/c_vgood;


%---------------------------------------------------------------------

% boolean array for maintainence
bool_maint_vhigh = strcmp('vhigh',train_data(:,2));
bool_maint_high = strcmp('high',train_data(:,2));
bool_maint_med = strcmp('med',train_data(:,2));
bool_maint_low = strcmp('low',train_data(:,2));

% conditional probability for buying
p_maint_vhigh_unacc = sum(and(bool_maint_vhigh,bool_unacc))/c_unacc;
p_maint_vhigh_acc = sum(and(bool_maint_vhigh,bool_acc))/c_acc;
p_maint_vhigh_good = sum(and(bool_maint_vhigh,bool_good))/c_good;
p_maint_vhigh_vgood = sum(and(bool_maint_vhigh,bool_vgood))/c_vgood;

p_maint_high_unacc = sum(and(bool_maint_high,bool_unacc))/c_unacc;
p_maint_high_acc = sum(and(bool_maint_high,bool_acc))/c_acc;
p_maint_high_good = sum(and(bool_maint_high,bool_good))/c_good;
p_maint_high_vgood = sum(and(bool_maint_high,bool_vgood))/c_vgood;

p_maint_med_unacc = sum(and(bool_maint_med,bool_unacc))/c_unacc;
p_maint_med_acc = sum(and(bool_maint_med,bool_acc))/c_acc;
p_maint_med_good = sum(and(bool_maint_med,bool_good))/c_good;
p_maint_med_vgood = sum(and(bool_maint_med,bool_vgood))/c_vgood;

p_maint_low_unacc = sum(and(bool_maint_low,bool_unacc))/c_unacc;
p_maint_low_acc = sum(and(bool_maint_low,bool_acc))/c_acc;
p_maint_low_good = sum(and(bool_maint_low,bool_good))/c_good;
p_maint_low_vgood = sum(and(bool_maint_low,bool_vgood))/c_vgood;

%----------------------------------------------------------------------

% boolean array for doors
bool_doors_2 = strcmp('2',train_data(:,3));
bool_doors_3 = strcmp('3',train_data(:,3));
bool_doors_4 = strcmp('4',train_data(:,3));
bool_doors_5more = strcmp('5more',train_data(:,3));


% conditional probability for doors

p_doors_2_unacc = sum(and(bool_doors_2,bool_unacc))/c_unacc;
p_doors_2_acc = sum(and(bool_doors_2,bool_acc))/c_acc;
p_doors_2_good = sum(and(bool_doors_2,bool_good))/c_good;
p_doors_2_vgood = sum(and(bool_doors_2,bool_vgood))/c_vgood;

p_doors_3_unacc = sum(and(bool_doors_3,bool_unacc))/c_unacc;
p_doors_3_acc = sum(and(bool_doors_3,bool_acc))/c_acc;
p_doors_3_good = sum(and(bool_doors_3,bool_good))/c_good;
p_doors_3_vgood = sum(and(bool_doors_3,bool_vgood))/c_vgood;

p_doors_4_unacc = sum(and(bool_doors_4,bool_unacc))/c_unacc;
p_doors_4_acc = sum(and(bool_doors_4,bool_acc))/c_acc;
p_doors_4_good = sum(and(bool_doors_4,bool_good))/c_good;
p_doors_4_vgood = sum(and(bool_doors_4,bool_vgood))/c_vgood;

p_doors_5more_unacc = sum(and(bool_doors_5more,bool_unacc))/c_unacc;
p_doors_5more_acc = sum(and(bool_doors_5more,bool_acc))/c_acc;
p_doors_5more_good = sum(and(bool_doors_5more,bool_good))/c_good;
p_doors_5more_vgood = sum(and(bool_doors_5more,bool_vgood))/c_vgood;



%----------------------------------------------------------------------

% boolean array for doors
bool_persons_2 = strcmp('2',train_data(:,4));
bool_persons_4 = strcmp('4',train_data(:,4));
bool_persons_more = strcmp('more',train_data(:,4));


% conditional probability for persons

p_persons_2_unacc = sum(and(bool_persons_2,bool_unacc))/c_unacc;
p_persons_2_acc = sum(and(bool_persons_2,bool_acc))/c_acc;
p_persons_2_good = sum(and(bool_persons_2,bool_good))/c_good;
p_persons_2_vgood = sum(and(bool_persons_2,bool_vgood))/c_vgood;

p_persons_4_unacc = sum(and(bool_persons_4,bool_unacc))/c_unacc;
p_persons_4_acc = sum(and(bool_persons_4,bool_acc))/c_acc;
p_persons_4_good = sum(and(bool_persons_4,bool_good))/c_good;
p_persons_4_vgood = sum(and(bool_persons_4,bool_vgood))/c_vgood;

p_persons_more_unacc = sum(and(bool_persons_more,bool_unacc))/c_unacc;
p_persons_more_acc = sum(and(bool_persons_more,bool_acc))/c_acc;
p_persons_more_good = sum(and(bool_persons_more,bool_good))/c_good;
p_persons_more_vgood = sum(and(bool_persons_more,bool_vgood))/c_vgood;



%----------------------------------------------------------------------
% lug_boot: small, med, big. 
% boolean array for lug_boot

bool_lug_boot_small = strcmp('small',train_data(:,5));
bool_lug_boot_med = strcmp('med',train_data(:,5));
bool_lug_boot_big = strcmp('big',train_data(:,5));

% conditional probability for lug boot

p_lug_boot_small_unacc = sum(and(bool_lug_boot_small,bool_unacc))/c_unacc;
p_lug_boot_small_acc = sum(and(bool_lug_boot_small,bool_acc))/c_acc;
p_lug_boot_small_good = sum(and(bool_lug_boot_small,bool_good))/c_good;
p_lug_boot_small_vgood = sum(and(bool_lug_boot_small,bool_vgood))/c_vgood;

p_lug_boot_med_unacc = sum(and(bool_lug_boot_med,bool_unacc))/c_unacc;
p_lug_boot_med_acc = sum(and(bool_lug_boot_med,bool_acc))/c_acc;
p_lug_boot_med_good = sum(and(bool_lug_boot_med,bool_good))/c_good;
p_lug_boot_med_vgood = sum(and(bool_lug_boot_med,bool_vgood))/c_vgood;

p_lug_boot_big_unacc = sum(and(bool_lug_boot_big,bool_unacc))/c_unacc;
p_lug_boot_big_acc = sum(and(bool_lug_boot_big,bool_acc))/c_acc;
p_lug_boot_big_good = sum(and(bool_lug_boot_big,bool_good))/c_good;
p_lug_boot_big_vgood = sum(and(bool_lug_boot_big,bool_vgood))/c_vgood;



%----------------------------------------------------------------------
% safety: low, med, high.
% boolean array for safety

bool_safety_low = strcmp('low',train_data(:,6));
bool_safety_med = strcmp('med',train_data(:,6));
bool_safety_high = strcmp('high',train_data(:,6));

% conditional probability for safety

p_safety_low_unacc = sum(and(bool_safety_low,bool_unacc))/c_unacc;
p_safety_low_acc = sum(and(bool_safety_low,bool_acc))/c_acc;
p_safety_low_good = sum(and(bool_safety_low,bool_good))/c_good;
p_safety_low_vgood = sum(and(bool_safety_low,bool_vgood))/c_vgood;

p_safety_med_unacc = sum(and(bool_safety_med,bool_unacc))/c_unacc;
p_safety_med_acc = sum(and(bool_safety_med,bool_acc))/c_acc;
p_safety_med_good = sum(and(bool_safety_med,bool_good))/c_good;
p_safety_med_vgood = sum(and(bool_safety_med,bool_vgood))/c_vgood;

p_safety_high_unacc = sum(and(bool_safety_high,bool_unacc))/c_unacc;
p_safety_high_acc = sum(and(bool_safety_high,bool_acc))/c_acc;
p_safety_high_good = sum(and(bool_safety_high,bool_good))/c_good;
p_safety_high_vgood = sum(and(bool_safety_high,bool_vgood))/c_vgood;



%------------------------------------------------------------------------
%------------------------------------------------------------------------
% nested function

    function resultClass = nestedGetClass(newData)
        
        resultClass = cell(length(newData),1);
        rowIndex = 1;

        for index = 1:length(newData)

            p_unacc=log1p(prior_unacc);
            p_acc=log1p(prior_acc);
            p_good=log1p(prior_good);
            p_vgood=log1p(prior_vgood);

            row = newData(index,:);

            %probability for buying
            % buying: vhigh, high, med, low. 
            switch(row{1})
                case 'vhigh'
                    p_unacc = p_unacc+checkVal(p_buying_vhigh_unacc);
                    p_acc = p_acc+checkVal(p_buying_vhigh_acc);
                    p_good = p_good+checkVal(p_buying_vhigh_good);
                    p_vgood = p_vgood+checkVal(p_buying_vhigh_vgood);
                case 'high'
                    p_unacc = p_unacc+checkVal(p_buying_high_unacc);
                    p_acc = p_acc+checkVal(p_buying_high_acc);
                    p_good = p_good+checkVal(p_buying_high_good);
                    p_vgood = p_vgood+checkVal(p_buying_high_vgood);
                case 'med'
                    p_unacc = p_unacc+checkVal(p_buying_med_unacc);
                    p_acc = p_acc+checkVal(p_buying_med_acc);
                    p_good = p_good+checkVal(p_buying_med_good);
                    p_vgood = p_vgood+checkVal(p_buying_med_vgood);
                case 'low'
                    p_unacc = p_unacc+checkVal(p_buying_low_unacc);
                    p_acc = p_acc+checkVal(p_buying_low_acc);
                    p_good = p_good+checkVal(p_buying_low_good);
                    p_vgood = p_vgood+checkVal(p_buying_low_vgood);
            end
            % maint: vhigh, high, med, low. 
             switch(row{2})
                 case 'vhigh'
                    p_unacc = p_unacc+checkVal(p_maint_vhigh_unacc);
                    p_acc = p_acc+checkVal(p_maint_vhigh_acc);
                    p_good = p_good+checkVal(p_maint_vhigh_good);
                    p_vgood = p_vgood+checkVal(p_maint_vhigh_vgood);
                 case 'high'
                     p_unacc = p_unacc+checkVal(p_maint_high_unacc);
                    p_acc = p_acc+checkVal(p_maint_high_acc);
                    p_good = p_good+checkVal(p_maint_high_good);
                    p_vgood = p_vgood+checkVal(p_maint_high_vgood);
                 case 'med'
                    p_unacc = p_unacc+checkVal(p_maint_med_unacc);
                    p_acc = p_acc+checkVal(p_maint_med_acc);
                    p_good = p_good+checkVal(p_maint_med_good);
                    p_vgood = p_vgood+checkVal(p_maint_med_vgood);
                 case 'low'
                    p_unacc = p_unacc+checkVal(p_maint_low_unacc);
                    p_acc = p_acc+checkVal(p_maint_low_acc);
                    p_good = p_good+checkVal(p_maint_low_good);
                    p_vgood = p_vgood+checkVal(p_maint_low_vgood);
             end

             % doors: 2, 3, 4, 5more. 
             switch(row{3})
                 case '2'
                    p_unacc = p_unacc+checkVal(p_doors_2_unacc);
                    p_acc = p_acc+checkVal(p_doors_2_acc);
                    p_good = p_good+checkVal(p_doors_2_good);
                    p_vgood = p_vgood+checkVal(p_doors_2_vgood);
                 case '3'
                    p_unacc = p_unacc+checkVal(p_doors_3_unacc);
                    p_acc = p_acc+checkVal(p_doors_3_acc);
                    p_good = p_good+checkVal(p_doors_3_good);
                    p_vgood = p_vgood+checkVal(p_doors_3_vgood);
                 case '4'
                    p_unacc = p_unacc+checkVal(p_doors_4_unacc);
                    p_acc = p_acc+checkVal(p_doors_4_acc);
                    p_good = p_good+checkVal(p_doors_4_good);
                    p_vgood = p_vgood+checkVal(p_doors_4_vgood);
                 case '5more'
                    p_unacc = p_unacc+checkVal(p_doors_5more_unacc);
                    p_acc = p_acc+checkVal(p_doors_5more_acc);
                    p_good = p_good+checkVal(p_doors_5more_good);
                    p_vgood = p_vgood+checkVal(p_doors_5more_vgood);
             end


             % persons: 2, 4, more. 
             switch(row{4})
                 case '2'
                    p_unacc = p_unacc+checkVal(p_persons_2_unacc);
                    p_acc = p_acc+checkVal(p_persons_2_acc);
                    p_good = p_good+checkVal(p_persons_2_good);
                    p_vgood = p_vgood+checkVal(p_persons_2_vgood);
                 case '4'
                    p_unacc = p_unacc+checkVal(p_persons_4_unacc);
                    p_acc = p_acc+checkVal(p_persons_4_acc);
                    p_good = p_good+checkVal(p_persons_4_good);
                    p_vgood = p_vgood+checkVal(p_persons_4_vgood);
                 case 'more'
                    p_unacc = p_unacc+checkVal(p_persons_more_unacc);
                    p_acc = p_acc+checkVal(p_persons_more_acc);
                    p_good = p_good+checkVal(p_persons_more_good);
                    p_vgood = p_vgood+checkVal(p_persons_more_vgood);         
             end

             % lug_boot: small, med, big.
             switch(row{5})
                 case 'small'
                    p_unacc = p_unacc+checkVal(p_lug_boot_small_unacc);
                    p_acc = p_acc+checkVal(p_lug_boot_small_acc);
                    p_good = p_good+checkVal(p_lug_boot_small_good);
                    p_vgood = p_vgood+checkVal(p_lug_boot_small_vgood);
                 case 'med'
                    p_unacc = p_unacc+checkVal(p_lug_boot_med_unacc);
                    p_acc = p_acc+checkVal(p_lug_boot_med_acc);
                    p_good = p_good+checkVal(p_lug_boot_med_good);
                    p_vgood = p_vgood+checkVal(p_lug_boot_med_vgood);
                 case 'big'
                    p_unacc = p_unacc+checkVal(p_lug_boot_big_unacc);
                    p_acc = p_acc+checkVal(p_lug_boot_big_acc);
                    p_good = p_good+checkVal(p_lug_boot_big_good);
                    p_vgood = p_vgood+checkVal(p_lug_boot_big_vgood);
             end          

            % safety: low, med, high. 

             switch(row{6})
                 case 'low'
                    p_unacc = p_unacc+checkVal(p_safety_low_unacc);
                    p_acc = p_acc+checkVal(p_safety_low_acc);
                    p_good = p_good+checkVal(p_safety_low_good);
                    p_vgood = p_vgood+checkVal(p_safety_low_vgood);
                 case 'med'
                    p_unacc = p_unacc+checkVal(p_safety_med_unacc);
                    p_acc = p_acc+checkVal(p_safety_med_acc);
                    p_good = p_good+checkVal(p_safety_med_good);
                    p_vgood = p_vgood+checkVal(p_safety_med_vgood);
                 case 'high'
                    p_unacc = p_unacc+checkVal(p_safety_high_unacc);
                    p_acc = p_acc+checkVal(p_safety_high_acc);
                    p_good = p_good+checkVal(p_safety_high_good);
                    p_vgood = p_vgood+checkVal(p_safety_high_vgood);
             end     

             if p_unacc>=p_acc && p_unacc>=p_good && p_unacc>=p_vgood
                 resultClass{rowIndex} = 'unacc';
             elseif p_acc>=p_unacc && p_acc>=p_good && p_acc>=p_vgood
                 resultClass{rowIndex} = 'acc';
             elseif p_good>=p_unacc && p_good>=p_acc && p_good>=p_vgood
                 resultClass{rowIndex} = 'good';
             else
                 resultClass{rowIndex} = 'vgood';         
             end             
             rowIndex = rowIndex +1;
        end
    end

    resultClass = nestedGetClass(new_data);
    new_accu = getAccuracy(resultClass,new_label)
    
    resultClass = nestedGetClass(train_data);
    train_accu = getAccuracy(resultClass,train_label)
end


function result = checkVal(val)
const = log1p(0.00000001);
if val == 0
    result = const;
else
    result = log1p(val);
end
end


function accuracy = getAccuracy(resultClass, newLabel)
sum = 0;
for index = 1:length(resultClass)
    if (strcmp(resultClass{index},newLabel{index}))
        sum = sum+1;
    end
end
accuracy = sum/length(resultClass);
end



