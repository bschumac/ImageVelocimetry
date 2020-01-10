pix = readtable('/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/Tb_stab_cut_red_3Hz_hardsubsample_Falsepixel.txt');
pix = table2array(pix);

fs = 3;

imfs = emd(pix);
imf1 = imfs(:,1)
imf2 = imfs(:,2)

[hs,f,t,imfinsf,imfinse] = hht(imf1,fs);


(sum(imfinsf.* imfinse/sum(imfinse)))/519
% gewichtetes arithmetisches mittel fuer imf1 und imf2
a = table(imfinsf, imfinse, imfinse/sum(imfinse), imfinsf.*imfinse/sum(imfinse))
= sum(imfinsf.*imfinse/sum(imfinse)) % 0.7359

b=table(imfinsf, imfinse, imfinse/sum(imfinse), imfinsf.*imfinse/sum(imfinse))
sum(imfinsf.*imfinse/sum(imfinse)) % 0.2419