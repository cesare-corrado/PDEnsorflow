"""
This is a gpuSolve package that implements functions to solve PDEs using tensorflow.

"""


__version__=['1','0','1']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)
    

#print(' This is gpuSolve Version {}'.format(__version__),flush=True)




