pix = readtable('/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/Tb_stab_cut_red_3Hz_hardsubsample_Falsepixel.txt');
pix = table2array(pix);
plot(pix)
fs = 3;

imfs = emd(pix);
imf1 = imfs(:,1);
imf2 = imfs(:,2);
imf2 = imfs(:,2);
imf2 = imfs(:,2);
imf5 = imfs(:,5);
imf6 = imfs(:,6);
plot(imfs)
IMF = imf1;
varargin = fs;


[IMF,fs,T,TD,F,FRange,FResol,MinThres,Method,FreqLoc,isTT,isNF] = parseAndValidateInputs(IMF, varargin);

test_f = [1,2,3,4,5,6];


sig_real = real(sig);

for i = 1:size(imf1,2)
    %switch Method
        % for future extension
        %case 'HT'
            sig = hilbert(imf1);
            energy = abs(sig).^2;
            phaseAngle = angle(sig);
    %end


    % compute instantaneous frequency using phase angle
    omega = gradient(unwrap(phaseAngle));
    
    % convert to Hz
    omega = fs/(2*pi)*omega;
    
    % find out index of the frequency
    omegaIdx = floor((omega-F(1))/FResol)+1;
    freqIdx(i,:) = omegaIdx(:,1)';
    
    % generate distribution
    insf(:,i) = omega;
    inse(:,i) = energy;
end

plot(inse)


[hs,f,t,imfinsf,imfinse] = hht(imf2,fs);
hht(imf1,fs);



(sum((imfinsf.* imfinse)/sum(imfinse)))/519
% gewichtetes arithmetisches mittel fuer imf1 und imf2
% => 

a = table(imfinsf, imfinse, imfinse/sum(imfinse), imfinsf.*imfinse/sum(imfinse))
sum((imfinsf.*imfinse)/sum(imfinse)) % 0.7359


b=table(imfinsf, imfinse, imfinse/sum(imfinse), imfinsf.*imfinse/sum(imfinse))
sum(imfinsf.*imfinse/sum(imfinse)) % 0.2419