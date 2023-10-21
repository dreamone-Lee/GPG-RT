# BENCHMARK
修改map路径、map_id后、test_name后直接运行  
需要申请gurobi的license

## experiment_benchmark
包含GP3、PQL、CTD、AC的实验

## experiment_benchmark2
包含FMA、DOT、OS-MIP、ILP的实验
map_id：设置0~7依次代表Sioux Falls、Anaheim、Winnipeg、Chicago-Sketch、Chengdu-Weekend Off-peak Hour、Chengdu-Weekend Peak Hour、Chengdu-Weekday Off-peak Hour、Chengdu-Weekday Peak Hour  
OD_pairs：保存在Networks目录下  
T_factor：T_{let}前乘上的系数(0.95,1,1.05)  
kappa：论文中生成方差的系数（0.15,0.25,0.5）  
K：FMA的最大迭代次数  
S：PLM、ILP、OS-MIP的采样数  
MaxIter：PLM、ILP、OS-MIP的最大迭代次数  
DOT_delta：DOT的窗口大小  



