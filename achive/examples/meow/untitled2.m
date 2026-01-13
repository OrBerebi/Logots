clc
clear
close all

% get the meow sounds
[m1,fs ]= audioread("110010__tuberatanka__cat-meow-ii.wav");
m2 = audioread("262314__steffcaffrey__cat-meow3.wav");
m3 = audioread("412017__skymary__cat-meow-short.wav");
m1 = m1(:,1);
m2 = m2(:,1);
m3 = m3(:,1);
% get the human sounds
[s1,fs1] = audioread("adult_female_speech.wav");
s1 = s1(:,1);

% resmaple the human sounds to match sample rate of meow
gComDiv = gcd(fs1, fs);
p = double(fs / gComDiv);
q = double(fs1 / gComDiv);
s1_resampled =resample(s1,p,q);

%preper out signal
out_length_sec = 13;
t_sig = linspace(0,out_length_sec,fs*out_length_sec).';
sig_out = zeros(size(t_sig));
zeros_start = linspace(0,2,fs*2).';
zeros_start = zeros(size(zeros_start));

meow_and_human = [zeros_start;m1;m2;m3;s1_resampled];


sig_out = meow_and_human(1:length(sig_out));

audiowrite("sig_out.wav",sig_out,fs);


%figure
%plot(sig_out)
%soundsc(sig_out,fs)