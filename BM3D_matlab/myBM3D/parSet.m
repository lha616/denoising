function par = parSet(sigma)
    if sigma < 40
        par.N1     = 8;     % block_matching 的block大小
        par.N2     = 16;    % 最相似块个数
        par.Ns     = 39;    % The size of block-matching neighborhood. 
        par.Nstep  = 3;
        par.Thr    = 2.7;
    else
        par.N1     = 8;     
        par.N2     = 32;    
        par.Ns     = 39;     
        par.Nstep  = 4;
        par.Thr    = 2.8;
    end
end

