 # polychoric correlation
 import Pkg
 using Pkg
 Pkg.add(url="https://github.com/nalimilan/FreqTables.jl")
 Pkg.add(url="https://github.com/PharmCat/MvNormalCDF.jl")
 Pkg.add("Distributions")
 using Distributions
 using MvNormalCDF
 using FreqTables
 Pkg.add("Optim")
 using Optim

 # define two helper functions 
 rowSums(x) = sum.(eachrow(x))
 colSums(x) = sum.(eachcol(x)) 
 
 
 
  x = [(i%7)%4 for i in 1:100]  # will be rows
  y = [i%3 for i in 1:100]  # will be columns

 num_r = length(unique(x))

  normal = Normal()

  function mvDistFunTable(rho, row_cuts, col_cuts) 
    num_r = length(row_cuts) - 1
    num_c = length(col_cuts) - 1

    # generate a bivariate correlation matrix with correlation rho
    μ0 = [0, 0]
    Σ0=[[1, rho] [rho, 1]]
    d = MvNormal(μ0, Σ0)

    # compute mass under rectangular supports
    sp = [ mvnormcdf(d, [row_cuts[i],col_cuts[j]],[row_cuts[i+1],col_cuts[j+1]])[1] for i in 1:num_r, j in 1:num_c ]

    # should sum up to 1 when bounds are +-Inf
    sp = sp ./ sum(sp)

    return sp
  end


  # Test
 mv_result= mvDistFunTable(.1, [-Inf, 0, 1, Inf],[-Inf, -1, Inf])
# sum(mv_result) should sum up to 1
  

  
  pars = [.1, -1,0,1, 2, -2,0,.5, Inf]
  pars = [.1, -1,0,1, 2, -2,0,-.5, Inf]

  #=
  this is the polychoric loss function
  for the inner optimization run
  
  arguments: an array of parameters of which the first
  is the correlation, followed by th row cuts, followed
  by the column cuts

  needs to have access to variable num_r !!
  =#
  function pcloss(pars)
     rho = min(max(-1, pars[1]), +1)
     
     row_cuts = pars[2:(num_r+1)]
     col_cuts = pars[(num_r+2):end]

    row_cuts[2:end] = [max(1e-7,x) for x in row_cuts[2:end]]
    col_cuts[2:end] = [max(1e-7,x) for x in col_cuts[2:end]]

     # recode row cuts, such that they are always ordered
     row_cuts[2:end] = row_cuts[2:end] .+ row_cuts[1]     
     col_cuts[2:end] = col_cuts[2:end] .+ col_cuts[1]

     row_cuts = vcat(-Inf, row_cuts, +Inf)
     col_cuts = vcat(-Inf, col_cuts, +Inf)

     # genrate cross-tabulated data
     tab = freqtable(x,y) 
     # compute probability masses
     #print(row_cuts, " AND ",col_cuts, "\n")
     P = mvDistFunTable(rho, row_cuts, col_cuts)
     P[P.<=0].=1e-30
     #print(P,"\n------\n")
     #print(log(P),"\n")
     display(P)
     #display(tab)
     # compute weighted loss function
     myloss = -sum(tab .* log(P)) 
     return myloss
  end


  pcloss([0.1, 2,4,5, 6,7])

  opt = optimize(pcloss, [0.1, -1,2,  -1,+2])
  
  function f(pars)
     tab = freqtable(x,y) # tabulated data
     num_r = size(tab,1) # number of rows
     num_c = size(tab,2) # number of columns
     n = sum(tab) # total sample size
     rc = quantile(normal, cumsum(rowSums(tab))/n) # row thresholds
     cc = quantile(normal, cumsum(colSums(tab))/n) # column thresholds
     # remove last elements
     pop!(rc)
     pop!(cc)

     start_rho = 0.0
     optimize(pcloss, [start_rho, rc, cc ])
   return (rc, cc)
  end
 
  f(nothing)
