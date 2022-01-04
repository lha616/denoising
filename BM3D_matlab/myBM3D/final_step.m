function [imout] = final_step(im,imout,sigma)

[M1,M2]=size(im); 
if sigma < 40
N1    = 8;     %% block_matching 的block大小
N2    = 16;    %% 最相似块个数
Ns    = 39;    %% 参考patch的领域大小 
Nstep = 3;
else
N1    = 11; 
N2    = 16;    
Ns    = 39;    
Nstep = 6;
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 创建2D块变换矩阵。
%%%%
[Tfor, Tinv]   = getTransfMatrix(N1, 'dct', 0); 
Tfor=single(Tfor);
Tinv=single(Tinv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 创建第三纬变换矩阵。
%%%%
hadper_trans_single_den = cell(1,1000);
inverse_hadper_trans_single_den = cell(1,1000);
for hpow = 0:ceil(log2(N2))
    h = 2^hpow;
    [Tfor3rd, Tinv3rd]   = getTransfMatrix(h, 'haar', 0);
    hadper_trans_single_den{h}         = single(Tfor3rd);
    inverse_hadper_trans_single_den{h} = single(Tinv3rd);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 构造Kaiser窗.

Wwin2D    = kaiser(N1, 2) * kaiser(N1, 2)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
group1=cell(1,15000);      
group2=cell(1,15000);
Coord=zeros(80000,4);     
weight = zeros(1,10000); 
for i=1:15000
    group1{i}=zeros(N1,N1,N2); %% Predefine Each Group.
    group2{i}=zeros(N1,N1,N2);
end

im_p1=single(padarray(im,[40 40],'symmetric','both')); 
im_p2=single(padarray(imout,[40 40],'symmetric','both'));
[M,N]=size(im_p1);                                                        
 x = zeros(1000,1);                            
 y = zeros(1000,1);                           

 l=1;
 c=1;
 
 for a = 41:Nstep:M1+40                              %三维块匹配变换图像去噪主要功能
    for b = 41:Nstep:M2+40
     
       k=1;                                    
       im_n1=im_p1(a:a+N1-1,b:b+N1-1);           
       im_n2=im_p2(a:a+N1-1,b:b+N1-1); 

                   group1{l}(:,:,k) = im_n1; 
                   group2{l}(:,:,k) = im_n2;
                   Coord(c,1)=l;               
                   Coord(c,2)=1;               
                   Coord(c,3)=a;               
                   Coord(c,4)=b;               
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
             for R=1:length(x)
               ux = im_p2(x(R):x(R)+N1-1,y(R):y(R)+N1-1);   
                dis1(R,3)=sum(sum((ux-im_n2).*(ux-im_n2))); %% 计算欧氏距离作为相似度 
                dis1(R,1)=x(R);                            %%记录patch对应坐标
                dis1(R,2)=y(R);                            
             end
             
              dis2=sortrows(dis1,3);                       %% patchs按相似度排序并抽取最相似的N2-1个
              
              for R=1:N2-1                                
                   ux1 = im_p1(dis2(R,1):dis2(R,1)+N1-1,dis2(R,2):dis2(R,2)+N1-1);
                   ux2 = im_p2(dis2(R,1):dis2(R,1)+N1-1,dis2(R,2):dis2(R,2)+N1-1);
                   group1{l}(:,:,k) = ux1;   
                   group2{l}(:,:,k) = ux2;
                   Coord(c,1)=l;                            
                   Coord(c,2)=k;
                   Coord(c,3)=dis2(R,1);                   
                   Coord(c,4)=dis2(R,2);                    
                   c=c+1;
                   k=k+1;
              end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
  %% 2D block 正向变换
  for K=1:N2
        group1{l}(:,:,K)=Tfor*group1{l}(:,:,K)*Tfor';
        group2{l}(:,:,K)=Tfor*group2{l}(:,:,K)*Tfor'; 
   end 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
 %% 第三维正向变换、硬阈值
 for j=1:N1
           
           d31=zeros(N2,N1); 
           d32=zeros(N2,N1); 

           for u=1:N2
                    d31(u,:)=group1{l}(:,j,u); 
                    d32(u,:)=group2{l}(:,j,u);
           end 

                 d31 = hadper_trans_single_den{h}*d31; 
                 d32 = hadper_trans_single_den{h}*d32;
                 
                 d31 = ((d32.^2)./(d32.^2+(sigma/255)^2)).*d31; 
                 % 维纳滤波
                 
           for u=1:N2
                group1{l}(:,j,u) =d31(u,:);    
           end 
           for u=1:N2
                group2{l}(:,j,u) =d32(u,:);    
           end   
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
 %% 计算每个group权重

    weight(l)=1/((norm((group2{l}(:).^2)./((group2{l}(:).^2)+(sigma/255)^2))^2)*(sigma/255)^2);
        
     
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
 %% 第三维逆变换。  
  for j=1:N1

           d31=zeros(N2,N1); 

           for u=1:N2
               d31(u,:)=group1{l}(:,j,u);
           end 

               d31=inverse_hadper_trans_single_den{h}*d31;
               
           for u=1:N2
                group1{l}(:,j,u) =d31(u,:);
           end   
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 %% 2D block 反变换.       
   for K=1:N2
        group1{l}(:,:,K)=Tinv*group1{l}(:,:,K)*Tinv';
   end 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
              l=l+1;  
                
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 最终聚合处理
imr=zeros(M,N); 
W=zeros(M,N);  

C=length(Coord(:,1)); 
for r=1:C
      if Coord(r,1)==0
         break;
      end
      %% Group 聚合
      imr(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)=imr(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)+group1{Coord(r,1)}(:,:,Coord(r,2))*weight(Coord(r,1)).*Wwin2D; 
      %% Weight 聚合
      W(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)=W(Coord(r,3):Coord(r,3)+N1-1,Coord(r,4):Coord(r,4)+N1-1)+weight(Coord(r,1))*Wwin2D; 
end


imr=imcrop(imr,[41 41 M2-1 M1-1]); 
W=imcrop(W,[41 41 M2-1 M1-1]);    

imout=imr./W;                              






 