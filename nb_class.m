function [mdl,classperformance,distribution_means,performance_table] = nb_class(X,Y,k)
%Naive Bayes Classifier with k-fold classification
%   X is N x M matrix of predictor data
%   Y is array of classes
%   k is value of k-fold

% % normalise data from 0 to 1
% normX = X - min(X(:));
% normX = normX ./ max(normX(:));

% generate indices for cross-validation
indices = crossvalind('Kfold',Y,k);

% initialize object to measure classifier performance
classperformance = classperf(Y);

% distribution for naive bayes
bayes_dist='normal';

% classify and report error rate (ratio of number of incorrectly classified samples/total number of classified samples)
for nk = 1:k
    % create indices for test/training set data
    test_idx = (indices == nk);
    train_idx = ~test_idx;
    % fit model
    % if class array is double
    if isa(Y,'double')==1
        mdl = fitcnb(X(train_idx,:),Y(train_idx),'DistributionNames',bayes_dist);
    % if class array is cell
    else
        mdl = fitcnb(X(train_idx,:),Y(train_idx),'ClassNames',unique(Y)','DistributionNames',bayes_dist)
    end
    predicted_class  = predict(mdl,X(test_idx,:));
    classperf(classperformance,predicted_class,test_idx);
end

% save distribution parameters
distribution_means=cell2mat(mdl.DistributionParameters);
distribution_means=distribution_means(1:2:end,:);
% put into table
distribution_means=array2table([NaN(size(unique(Y),1),1),distribution_means]);
distribution_means.Properties.VariableNames(1)={'Class'};
% add variable names
for i=1:width(distribution_means)-1
    distribution_means.Properties.VariableNames(i+1)={['Var',num2str(i)]};
end
distribution_means.Class=unique(Y);

% performance table
accuracy=1-classperformance.ErrorRate;
sensitivity=classperformance.Sensitivity;
specificity=classperformance.Specificity;
performance_table=table(accuracy,sensitivity,specificity,'VariableNames',{'Accuracy','Sensitivity','Specificity'});
end