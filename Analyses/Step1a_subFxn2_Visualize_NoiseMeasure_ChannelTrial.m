function [noise_measure_sorted] = Visualize_NoiseMeasure_ChannelTrial(noise_measure,plot_recordings)
%Takes any kind of noise measure computed across channels or trials (e.g.
%variance, mean absolute value) and sorts this array in descending order
%(to enable identification of high noise channels/trials), and plots the
%result
%plot_recordings: determines how many channels/trials to visualize in plot

%sort in ascending order
[noise_measure_sorted,i] = sort(noise_measure,'descend');
noise_measure_sorted = [noise_measure_sorted,i];

%plot - with circles for each recording
figure; hold on;
plot(noise_measure_sorted(1:plot_recordings,1),'-o'); %plot only 50 to help visualization
set(gca,'xticklabel',num2str(noise_measure_sorted(1:plot_recordings,2)));
set(gca,'xtick',1:plot_recordings);


end

