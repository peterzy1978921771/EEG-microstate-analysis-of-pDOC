
%% 子函数 SVMcgForClass.m
function [cg] = SVMcgForClass(train_label,train,c,g,v,bestnum)



cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(bestnum^c),' -g ',num2str(bestnum^g)];
cg = svmtrain(train_label, train, cmd); % cg相当于mse

