import numpy as np

bits=200
RR=RealField(bits)

def to_RR(arr):
    return np.array([RR(x) for x in arr])

def mul(mat,vec,axis_list):
    dd=mat.ndim
    size=len(vec)
    arr=mat
    for ax in axis_list:
        li=[1]*dd
        li[ax]=size
        sh=tuple(li)
        arr=arr*vec.reshape(sh)
    return arr

def block_sum(mat,block,axis_list):
    arr=mat
    for ax in axis_list:
        ra=np.arange(0,mat.shape[ax],block)
        arr=np.add.reduceat(arr,ra,axis=ax)
    return arr

class RSBbound:
    ca=1

    def __init__(self,deg,nrs):        
        self.deg=deg
        self.nrs=nrs
        self.depth=len(nrs)
        self.nr_prod=np.prod(nrs)
    
    #cf: factor constant (star: la-1 ;edge: 1/la-1)
    def func(self,dim,cf,ms,prs):
        axes=list(range(dim))

        pr=1-prs[-1] if dim==2 else prs[-1]
        mat=np.full(tuple([self.nr_prod]*dim),cf)
        mat=self.ca+mul(mat,pr,axes)

        for i in range(self.depth-1,-1,-1):
            mat=mat^(ms[i])
            mat=mul(mat,prs[i],axes)
            mat=block_sum(mat,self.nrs[i],axes)            

        return log(mat.flatten()[0])

    
    def bound(self,la,ms,prs):
        star=self.func(self.deg,la-1,ms,prs)
        edge=self.func(2,1/la-1,ms,prs)
        m_prod=np.prod(ms)
        return (star-self.deg*edge/2)/(m_prod*log(la)) 

def loadEx(id_nr):
    ex=load('ex'+str(id_nr)+'.sobj')
    bnd=RSBbound(ex['deg'],ex['nrs'])
    la=RR(ex['la'])
    ms=to_RR(ex['ms'])
    prs_or=ex['prs']
    prs=[to_RR(pr) for pr in prs_or]
    print(ex)
    print(bnd.bound(la,ms,prs))
    
def saveEx(id_nr):
    res=find_result(id_nr)
    deg=res['deg']
    nrs=res['nrs']
    arr=res['arr']
    rsbwr=RSBwrapper(deg,nrs)
    args=rsbwr.arr2args(arr)
    obj={'deg':deg, 'nrs':nrs, 'la':args[0], 'ms':args[1], 'prs':args[2], 'id_nr': id_nr, 'bound': res['bound']}
    save(obj,'ex'+str(id_nr)+'.sobj')
    print('example saved:')
    print(obj)
    
#loadEx(21094817)

#0.311922226926807499248433523176817639510936692072353775143851207749677260029385126215502542}
print(0.31192222692680749924785902438711925401064991042735564333464)
