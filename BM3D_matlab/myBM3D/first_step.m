function [imout] = first_step(im,sigma)

[M1,M2]=size(im); % 图像大小
if sigma < 40
N1     = 8;     % block_matching 的block大小
N2     = 16;    % 最相似块个数
Ns     = 39;    % 参考patch的领域大小
Nstep  = 3;
Thr    = 2.7;
else
N1     = 16;     
N2     = 32;    
Ns     = 39;     
Nstep  = 4;
Thr    = 2.8;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 创建2D块变换矩阵。
%%%%
[Tfor, Tinv]   = getTransfMatrix(N1, 'bior1.5', 0);     % 获取正变换和逆变换的矩阵
Tfor=single(Tfor);
Tinv=single(Tinv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 创建第三纬变换矩阵。
%%%%
hadper_trans_single_den = cell(1,1000);
inverse_hadper_trans_single_den = cell(1,1000);
for hpow = 0:ceil(log2(N2))
    h = 2^hpow;
    [Tfor3rd, Tinv3rd]  = getTransfMatrix(h, 'haar', 0);
    hadper_trans_single_den{h}  = single(Tfor3rd);
    inverse_hadper_trans_single_den{h} = single(Tinv3rd);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 构造Kaiser窗.

Wwin2D  = kaiser(N1, 2) * kaiser(N1, 2)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
group = cell(1,15000);      %% 预定义所有组
Coord = zeros(80000,4);     %% 预定义block坐标
weight = zeros(1,10000);  % 预定义权重

for i=1:15000
    group{i} = zeros(N1,N1,N2); %预定义每个组，即 3D 数组。
end


im_p = single(padarray(im,[40 40],'symmetric','both'));  %% 对称填充
[M,N] = size(im_p);                              %% 获取对称填充后size                             
 x = zeros(1000,1);                            %% 存储每组中的图像块水平坐标。
 y = zeros(1000,1);                            %% 存储每组中的图像块垂直坐标。

 L=1;
 c=1;
 
 for a = 41:Nstep:M1+40                              %三维块匹配变换图像去噪主要功能
    for b = 41:Nstep:M2+40
     
       k=1;                                    %% 1. 图像patch的 index.     
       im_n=im_p(a:a+N1-1,b:b+N1-1);           %% 1. im_n是参考patch，别的patches与im_n计算相似度

                   group{L}(:,:,k) = im_n;     %% 第L组第k块
                   Coord(c,1) = L;             %% 第 1 列代表数组 Coord 中的第 L 个组。
                   Coord(c,2) = k;             %% 第 2 列代表数组 Coord 中的第 k 个块
                   Coord(c,3) = a;             %% 第 3 列代表数组 Coord 中的水平坐标.
                   Coord(c,4) = b;             %% 第 4 列代表数组 Coord 中的垂直坐标。
                   c = c+1; 
                   k = k+1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% 创建块匹配邻域，它的邻域有 Ns*Ns-1 个patchs.将邻域patch的坐标填到x，y里

               
                Q = 1;
                for r = 1:(Ns-1)/2  %% (Ns-1)/2 是邻域半径  
                      for i = -r:r-1
                        x(Q) = a-r;
                        y(Q) = b+i;
                        Q = Q+1;
                      end   
                      for i = -r:r-1
                        x(Q) = a+i;
                        y(Q) = b+r;
                        Q = Q+1;
                      end   
                      for i = -r:r-1
                        x(Q) = a+r;
                        y(Q) = b-i;
                        Q = Q+1;
                      end   
                      for i = -r:r-1
                        x(Q) = a-i;
                        y(Q) = b-r;
                        Q = Q+1;
                      end 
                end
              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Block Matching 过程
             dis1=zeros(Ns^2-1,3);
             for R=1:length(x)                             %%遍历邻域
               ux = im_p(x(R):x(R)+N1-1,y(R):y(R)+N1-1);   %% 选择一个领域上的patch去做匹配
                dis1(R,3)=sum(sum((ux-im_n).*(ux-im_n)));  %% 计算参考块和要匹配的块之间的欧几里德距离。
                dis1(R,1)=x(R);                            %% 存储块水平坐标。
                dis1(R,2)=y(R);                            %% 存储块垂直坐标。 
             end
             
              dis2=sortrows(dis1,3);                       %% 按第 3 列索引对数组 dis1 进行排序，即欧氏距离。
              
              for R=1:N2-1                                 %% 选择与参考块最相似的15（N2 相似块个数=16） 个块并将它们作为第二个块到第 16 个块存储到组中。
                   ux = im_p(dis2(R,1):dis2(R,1)+N1-1,dis2(R,2):dis2(R,2)+N1-1);
                   group{L}(:,:,k) = ux;                    
                   Coord(c,1)=L;                            %% 第L组
                   Coord(c,2)=k;                            %% 第k个
                   Coord(c,3)=dis2(R,1);                    %% 第 3 列代表数组坐标中匹配块的水平坐标。
                   Coord(c,4)=dis2(R,2);                    %% 第 4 列表示数组坐标中匹配块的垂直坐标。
                   c=c+1;
                   k=k+1;
              end
              % 这一步将获得参考patch周围最相似的N2个patchs，记录了他们的坐标x,y
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
  %% 2D block 正向变换
  for K=1:N2
        group{L}(:,:,K)=Tfor*group{L}(:,:,K)*Tfor';
   end 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
 %% 第三维正向变换、硬阈值和逆变换。
 
 Non_zero = 0;
 
 for j=1:N1
    for u=1:N1
                 d3=zeros(N2,1); 
               
                 d3(:,1)=group{L}(j,u,:); %% group中的每一列都分配给一个临时列向量 d3。
           

                 d3 = hadper_trans_single_den{h}*d3; %% 第三维正向变换。
                 
                 d3 = d3.*(abs(d3)>=Thr*(sigma/255)); %% 硬阈值。小于阈值的置0
                 Non_zero = Non_zero + nnz(d3);
                 d3 = inverse_hadper_trans_single_den{h}*d3; %% 第三维逆变换。

                group{L}(j,u,:) =d3(:,1);  %% 转换后的列向量被替换为 group  
    end   
 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 %% 计算每个group权重

if Non_zero >= 1
    weight(L) = 1/Non_zero;
else
     weight(L)=1;  
end  
    

 %% 2D block 反变换.      
   for K=1:N2
        group{L}(:,:,K)=Tinv*group{L}(:,:,K)*Tinv';
   end 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
              L = L+1;  %% 计算下一组
                
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 最终聚合处理
imr=zeros(M,N); %% 创建与扩展图像相同大小的全零数组。
W=zeros(M,N);   %% 创建与扩展图像相同大小的全零数组作为权重数组。
C=length(Coord(:,1));  %% 检查组数。
for r=1:C
      if Coord(r,1)==0
         break;
      end
      %% Group 聚合
      imr(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)=imr(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)+group{Coord(r,1)}(:,:,Coord(r,2))*weight(Coord(r,1)).*Wwin2D; 
      %% Weight 聚合
      W(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)=W(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)+weight(Coord(r,1)).*Wwin2D; 
end


imr=imcrop(imr,[41 41 M2-1 M1-1]); %% 从扩展图像中裁剪有效部分。
W=imcrop(W,[41 41 M2-1 M1-1]);     %% 从扩展权重数组中裁剪有效部分。

imout=imr./W;                      %% 重复像素的平均值。                


