import numpy as np

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

def partitions(n,k):
    if k==1:
        return [[n]]
    if n==0:
        return [[0]*k]
    res=[]
    for first in range(n+1):
        subres=partitions(n-first,k-1)
        for part in subres:
            res.append([first]+part)
    return res

#assert sum(multinomial(*part) for part in partitions(13,6))==6^13
#print(max(partitions(18,4)))

def powers(arr,upto):
    pwr=[np.ones_like(arr)]
    for _ in range(upto):
        pwr.append(pwr[-1]*arr)
    return np.transpose(np.array(pwr))

def arr2str(arr,digits=5,sep=" "):
    str=""
    for val in arr:
        if str!="":
            str+=sep
        str+="{:.{}f}".format(val,digits)
    return str
			
class Result:
    ca=1

    def __init__(self,res):        
        self.deg=res['deg']
        self.nrs=res['nrs']
        self.depth=len(self.nrs)
        self.nr_prod=np.prod(self.nrs)
        self.la=res['la']
        self.ms=res['ms']
        self.prs=res['prs']
        self.stored_bnd=res['bound']
    
    #cf: factor constant (star: la-1 ;edge: 1/la-1)
    def func(self,dim,cf):
        axes=list(range(dim))

        pr=1-self.prs[-1] if dim==2 else self.prs[-1]
        mat=np.full(tuple([self.nr_prod]*dim),cf)
        mat=self.ca+mul(mat,pr,axes)

        for i in range(self.depth-1,-1,-1):
            mat=mat^(self.ms[i])
            mat=mul(mat,self.prs[i],axes)
            mat=block_sum(mat,self.nrs[i],axes)            

        return log(mat.flatten()[0])
   
    def bound(self):
        if self.depth==1 and self.deg>6:
            return self.twoRSBbound()
        star=self.func(self.deg,self.la-1)
        edge=self.func(2,1/self.la-1)
        m_prod=np.prod(self.ms)
        return (star-self.deg*edge/2)/(m_prod*log(self.la))

    def twoRSBbound(self):
        assert self.depth==1
        num=self.nrs[0]
        prtns=partitions(self.deg,num)
        parr=np.array(prtns,int)
        coeff=np.array([multinomial(*part) for part in prtns],int)
    
        m,p0,p1=self.ms[0],self.prs[0],self.prs[1]    
        p0_powers=powers(p0,self.deg)
        p1_powers=powers(p1,self.deg)
                
        def product(powers,part):
            prod=1
            for i in range(num):
                prod*=powers[i,part[i]]
            return prod
            
        prod0=np.array([product(p0_powers,part) for part in parr])
        prod1=np.array([product(p1_powers,part) for part in parr])
        star=np.dot(coeff,prod0*(1+(self.la-1)*prod1)**m)
        
        p1f=1-p1
        mat=(1+(1/self.la-1)*p1f.reshape((1,-1))*p1f.reshape((-1,1)))^m
        edge=np.dot(np.dot(mat,p0),p0)
                
        return (log(star)-self.deg/2*log(edge))/(m*log(self.la))
    
    def normalize(self):
        assert self.la>1
        for m in self.ms:
            assert m>0
            assert m<1
        for pr in self.prs:
            for p in pr:
                assert p>=0
                assert p<=1

        for k in range(self.depth):
            pr=self.prs[k]
            nr=self.nrs[k]
            ra=np.arange(0,len(pr),nr)
            prsum=np.add.reduceat(pr,ra)
            pr=pr/np.repeat(prsum,nr)
            self.prs[k]=pr
            #print(np.add.reduceat(pr,ra))
            
    def setPrecision(self,bits=200):
        RR=RealField(bits)
        self.la=RR(self.la)
        self.ms=np.array([RR(x) for x in self.ms])
        self.prs=[np.array([RR(x) for x in pr]) for pr in self.prs]
        self.normalize()

    def computeLayers(self):
        layp=[np.array([1.])]
        for k in range(self.depth):
            layp.append( self.prs[k]*np.repeat(layp[-1],self.nrs[k]) )
        layq=[self.prs[-1]]
        for k in range(self.depth-1,-1,-1):
            ra=np.arange(0,len(layq[-1]),self.nrs[k])
            layq.append(np.add.reduceat(self.prs[k]*layq[-1],ra))
        layq.reverse()
        self.layp=layp
        self.layq=layq
        #drawLayer(layp[-1],layq[-1])

    def drawLayer(self,k=-1):
        self.computeLayers()
        if k==-1:
            k=self.depth
        nrs=self.nrs[:k]
        ps=self.layp[k]
        qs=self.layq[k]
        nr_prod=len(ps)
        clrs=rainbow(nr_prod,'rgbtuple')
        fig=Graphics()
        for l in range(1,k):
            psum=0
            for p in self.layp[l][:-1]:
                psum+=p
                fig+=line( [(0,psum),(1,psum)] , rgbcolor=(0,0,0), thickness=k-l )

        psum=0.
        for i in range(nr_prod):
            fig+=line( [(qs[i],psum),(qs[i],psum+ps[i])] , rgbcolor=clrs[i], thickness=3)
            psum+=ps[i]

            
        fig.show(axes=False)

    def printLayer(self,k=-1):
        self.computeLayers()
        if k==-1:
            k=self.depth
        self.computeLayers()
        nrs=self.nrs[:k]
        ps=self.layp[k]
        qs=self.layq[k]
        nr_prod=len(ps)
        print("Atoms at layer {}:".format(k))
        for i in range(nr_prod):
            print("{:.8f} of size {:.5f}".format(qs[i],ps[i]))
            l=k-1
            ii=i+1
            while l>=0:
                if ii % nrs[l] == 0:
                    print("-----------------------")
                    ii/=nrs[l]
                    l-=1
                else:
                    l=-1            

    def print(self):		
        print("deg: {}".format(self.deg))
        print("r:   {}".format(self.depth+1))
        print("nrs: "+arr2str(self.nrs,0,"x"))
        print("la:  {:.2f}".format(self.la))
        print("ms:  "+arr2str(self.ms,3))
        k=0
        for pr in self.prs:
            k+=1
            pre="pr{}: ".format(k) if k<len(self.prs) else "1-q: "
            print(pre+arr2str(pr))
        print("bnd: {:.50f} (stored bound)".format(self.stored_bnd))

      
    
#def saveEx(id_nr):
#    res=find_result(id_nr)
#    deg=res['deg']
#    nrs=res['nrs']
#    arr=res['arr']
#    rsbwr=RSBwrapper(deg,nrs)
#    args=rsbwr.arr2args(arr)
#    obj={'deg':deg, 'nrs':nrs, 'la':args[0], 'ms':args[1], 'prs':args[2], 'id_nr': id_nr, 'bound': res['bound']}
#    save(obj,'ex'+str(id_nr)+'.sobj')
#    print('example saved:')
#    print(obj)

#id=13
#exs=load('examples.sobj')
#res=Result(exs[id])
#res.drawLayer()