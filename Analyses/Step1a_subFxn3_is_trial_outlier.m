function is_outlier = is_trial_outlier(data)
%% determines if a trial is an outlier 
    % Define threshold for outlier detection
    zscore_threshold = 3;
       
    % Outlier check: Z-score threshold
    zdata = zscore(data, 0, 2);  % z-score across time for each channel
    if any(any(abs(zdata) > zscore_threshold, 2))
        is_outlier = true;
        return;
    end
        
    is_outlier = false;
    return;
end
