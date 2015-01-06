clear all;

fid0 = fopen('result.txt','at'); %'wt'
classNum = 567;

for p=11:13
    fprintf(fid0,'P%f as test sample.\n',p);
    for i=0.1:0.1:1.0
        disp(p);
        disp(i);
        [rank1, rank5, rank10] = merge(p,i,classNum);
        fprintf(fid0,'%f\t%f\t%f\t%f\n',i,rank1,rank5,rank10);
    end

end
 fclose(fid0);