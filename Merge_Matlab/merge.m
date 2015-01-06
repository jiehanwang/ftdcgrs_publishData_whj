function [rank1, rank5, rank10] = merge(i,weight,totalNum)
str_C= strcat ('.\posture\result_detail_', int2str(i) , '.txt') ;
str_T= strcat ('.\trajectory\result_detail_', int2str(i) , '.txt') ;
result_detail_C = load(str_C);
result_detail_T = load(str_T);

result = zeros(totalNum,totalNum);
for i=1:totalNum
    for j=1:totalNum
        result(i,j) = weight*result_detail_C(i,j) + (1-weight)*result_detail_T(i,j);
    end
end

correct = 0;
for i=1:totalNum
    temp = find(result(i,:) == max(result(i,:)));
    if temp == i
        correct = correct + 1;
    end
end
rate = correct/totalNum;


rank1 = 0;
rank5 = 0;
rank10 = 0;
for i=1:totalNum
    [sA,index] = sort(result(i,:)); %从小到大排列
    sA = fliplr(sA);                %倒一下，变成从大到小排列
    index = fliplr(index);
    for k=1:10
        if k ==1 && index(k) == i
            rank1 = rank1+1;
        end
        if k<=5 && index(k) == i
            rank5 = rank5+1;
        end
        if k<=10 && index(k) == i
            rank10 = rank10 +1;
        end
    end
end

rank1 = rank1/totalNum;
rank5 = rank5/totalNum;
rank10 = rank10/totalNum;
