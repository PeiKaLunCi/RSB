import numpy as np
import random as rndm
from scipy import optimize
from statistics import median
import time

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
            
def compare(v1,v2,mes):
    print(v1,"COMPARISION FOR",mes)
    print(v2,"DIFF:", abs(v1-v2))
    
def depth_reduction(nrs,arr,reps=1):
    if reps==0:
        return nrs,arr
    
    depth=len(nrs)
    nr0=nrs[0]
    new_nrs=nrs[1:].copy()
    new_nrs[0]*=nr0
    ind=1+depth+nr0
    new_pr0=arr[ind:ind+new_nrs[0]]*np.repeat(arr[ind-nr0:ind],nrs[1],axis=0)
    return depth_reduction(new_nrs,np.concatenate([arr[0:1],arr[2:depth+1],new_pr0,arr[ind+new_nrs[0]:]]),reps-1)

def powers(arr,upto):
    pwr=[np.ones_like(arr)]
    for _ in range(upto):
        pwr.append(pwr[-1]*arr)
    return np.transpose(np.array(pwr))

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

#assert sum(multinomial(*part) for part in partitions(18,4))==4^18
#print(max(partitions(18,4)))

class RSBwrapper:
    rel_step={'la':1./128,'m':1./4,'pr':1.}    

    def __init__(self,deg,nrs,comp_class=None,bits=0,float_output=True):
        if comp_class is None:
            comp_class=RSBbound
        self.comp=comp_class(deg,nrs)
        self.deg=deg
        self.nrs=nrs
        self.depth=len(nrs)
        self.nr_prods=[np.prod(nrs[:j+1]) for j in range(self.depth)]
        self.pr_length=sum(self.nr_prods)+self.nr_prods[-1]
        self.total_length=1+self.depth+self.pr_length

        #predefined slices
        ind=self.depth+1
        self.m_slice=slice(1,ind)
        self.pr_slices=[]
        for j in range(self.depth):
            self.pr_slices.append(slice(ind,ind+self.nr_prods[j]))
            ind+=self.nr_prods[j]
        self.pr_slices.append(slice(ind,None,None))
        
        #self.vector_slices=[[] for _ in range(self.depth)]
        self.vector_slices=[]
        ind=self.depth+1  
        for j in range(self.depth):
            for i in range(0,self.nr_prods[j],self.nrs[j]):
                #self.vector_slices[j].append(slice(ind+i,ind+i+self.nrs[j]))
                self.vector_slices.append(slice(ind+i,ind+i+self.nrs[j]))
            ind+=self.nr_prods[j]    

        self.precise=(bits>0)
        if self.precise:
            self.RR=RealField(bits)
            self.float_output=float_output
        else:
            self.float_output=False

        #penalizing factor (for sums over vector_slices)
        self.offset_constant=1
        #distortion parameters (for forcing m to stay away from 1)
        self.m_distortion=False
        self.m_distortion_limit=0.92
        self.m_distortion_constant=2.^-13
        #what kind of perturbation to do
        self.perturbation_mode='default'
        #self.perturbation_mode='prlast'

        #an R->(0,1) bijection and its inverse and its derivative
        self.bij=lambda x:1/(1+exp(-x))
        self.vfunc=np.vectorize(self.bij)
        self.vfuncinv=np.vectorize(lambda x:-log(1/x-1))
        self.vder=np.vectorize(lambda x:exp(-x)/(exp(-x)+1)^2)
        #another bijection that is not working as well...
        #self.vfunc=np.vectorize(lambda x:(x+sqrt(x^2+4)-2)/(2*x))
        #self.vfuncinv=np.vectorize(lambda x:(2*x-1)/(x*(1-x)))
        #self.vder=np.vectorize(lambda x:1/2*(x/sqrt(x^2 + 4) + 1)/x - 1/2*(x + sqrt(x^2 + 4) - 2)/x^2)
        
        #default setup for rand
        la_est=17+0.5*self.deg+3*self.depth
        self.la_bounds=(la_est-1,la_est+1)
        self.m_bounds=(0.99,0.99)
        
        #callback settings for optimize.minimize
        self.callback_enabled=False
        self.callback_block=128
        self.callback_count=0
        self.callback_stop=lambda iters,arr: False

    def to_RR(self,arr,bits=None):
        RR=self.RR if bits==None else RealField(bits)
        return np.array([RR(x) for x in arr])

    def arr2args(self,arr):        
        return arr[0],arr[self.m_slice],[arr[sl] for sl in self.pr_slices]
    
    def prearr2arr(self,prearr):
        arr=self.vfunc(prearr)
        arr[0]=1+prearr[0]**2
        #arr[0]=1/arr[0]
        return arr
        
    def arr2prearr(self,arr):        
        aux=arr[0]
        arr[0]=0.5
        #arr[0]=1/aux
        prearr=self.vfuncinv(arr)
        prearr[0]=sqrt(aux-1)
        arr[0]=aux
        return prearr

    def normalize(self,arr):    
        for sl in self.vector_slices:
            arr[sl]/=arr[sl].sum()
    
    def get_normalized(self,arr):
        arr_n=arr.copy()
        self.normalize(arr_n)
        return arr_n

    def vector_sums(self,arr):
        return [arr[sl].sum()-1 for sl in self.vector_slices]
    
    def fu(self,prearr):
        arr=self.prearr2arr(prearr)
        fu_extra=self.offset_constant*sum((arr[sl].sum()-1)^2 for sl in self.vector_slices)
        if self.m_distortion:
            for m in arr[self.m_slice]:
                x=m-self.m_distortion_limit
                if x>0:
                    fu_extra+=self.m_distortion_constant*x^2
                    
        return self.fu_arr(arr)+fu_extra

    def der(self,prearr):
        arr=self.prearr2arr(prearr)
        der0=self.der_arr(arr)
        for sl in self.vector_slices:
            der0[sl]+=2*self.offset_constant*(arr[sl].sum()-1)
        if self.m_distortion:
            for i in range(1,1+self.depth):
                x=arr[i]-self.m_distortion_limit
                if x>0:
                    der0[i]+=2*self.m_distortion_constant*x
            
        pre_factor=self.vder(prearr)
        pre_factor[0]=2*prearr[0]
        #pre_factor[0]*=(-1/self.bij(prearr[0])^2)
        return pre_factor*der0

    def fu_arr(self,arr):
        if self.precise:
            arr=self.to_RR(arr)
        args=self.arr2args(arr)
        res=self.comp.bound(*args)
        if self.float_output:
            return float(res)
        else:
            return res
        
    def der_arr(self,arr):
        if self.precise:
            arr=self.to_RR(arr)
        args=self.arr2args(arr)
        res=self.comp.bound_der(*args)            
        if self.float_output:
            return res.astype(float)
        else:
            return res
    
    def fu_arr_precise(self,arr,bits):
        arr_precise=self.to_RR(arr,bits)
        self.normalize(arr_precise)
        args=self.arr2args(arr_precise)
        return self.comp.bound(*args)

    def rand(self,la_bounds=None,m_bounds=None,prearr_output=False):
        if la_bounds==None:
            la_bounds=self.la_bounds
        if m_bounds==None:
            m_bounds=self.m_bounds
        la=np.random.rand(1)*(la_bounds[1]-la_bounds[0])+la_bounds[0]
        m=np.random.rand(self.depth)*(m_bounds[1]-m_bounds[0])+m_bounds[0]
        pr=np.random.rand(self.pr_length)
        arr=np.concatenate([la,m,pr])
        self.normalize(arr)
        if prearr_output:
            arr=self.arr2prearr(arr)
        return arr
        
    def print(self,arr_in,what={'la','m','pr'},upto=100,prearr_input=False):
        arr=self.prearr2arr(arr_in) if prearr_input else arr_in
        if 'la' in what:
            print("la=",arr[0])
        if 'm' in what:    
            print("m=",arr[self.m_slice])
        if 'pr' in what:
            for j in range(min(upto,self.depth)+1):
                sh=tuple(self.nrs[:j+1])
                print('pr'+str(j)+'=',arr[self.pr_slices[j]].reshape(sh))        
    
    def print_parameters(self):
        print('rel_step=',self.rel_step)
        print('perturbation_mode=',self.perturbation_mode)
        print('offset_constant=',self.offset_constant)
        print('rand la&m bounds=',self.la_bounds,self.m_bounds)
        print('m distortion=',self.m_distortion,self.m_distortion_limit,self.m_distortion_constant)
    
    def duplicate(self,arr,dep):
        assert dep>=0 and dep<self.depth
        new_arr_list=[arr[:self.pr_slices[dep].start]]
        aux=arr[self.pr_slices[dep]].copy().reshape((-1,self.nrs[dep]))
        maxind=np.argmax(aux,axis=1)
        num=len(maxind)
        arr_dup=np.zeros((num,1))
        for i in range(num):
            maxval=aux[i,maxind[i]]
            aux[i,maxind[i]]=maxval/2
            arr_dup[i,0]=maxval/2
        new_arr_list.append(np.concatenate((aux,arr_dup),axis=1).flatten())
        for j in range(dep+1,self.depth+1):
            aux=arr[self.pr_slices[j]].reshape((num,self.nrs[dep],-1))
            arr_dup=np.zeros((aux.shape[0],1,aux.shape[2]))
            for i in range(num):
                arr_dup[i,0]=aux[i,maxind[i],:]
            new_arr_list.append(np.concatenate((aux,arr_dup),axis=1).flatten())
        return np.concatenate(new_arr_list)    
    
    def duplicate_iter(self,arr,dep,iters):
        new_arr=self.duplicate(arr,dep)
        if iters==1:
            return new_arr
        nrs=self.nrs.copy()
        nrs[dep]+=1
        return RSBwrapper(self.deg,nrs).duplicate_iter(new_arr,dep,iters-1)

    def perturbe(self,prearr,step):
        if self.perturbation_mode=='prlast':
            return self.perturbe_prlast(prearr,step)
        step_la=step*self.rel_step['la']
        step_m=step*self.rel_step['m']
        step_pr=step*self.rel_step['pr']
        st=np.array([step_la]+[step_m]*self.depth+[step_pr]*self.pr_length)
        rnd=2*np.random.rand(self.total_length)-1
        new_prearr=prearr+rnd*st
        arr=self.prearr2arr(new_prearr)
        self.normalize(arr)        
        return self.arr2prearr(arr)
    
    def perturbe_prlast(self,prearr,step):
        new_prearr=prearr.copy()
        rnd=step*(2*np.random.rand(self.nr_prods[-1])-1)
        new_prearr[self.pr_slices[-1]]+=rnd
        return new_prearr        

    def callback(self,x):
        if self.callback_enabled:
            self.callback_count+=1
            if self.callback_count%self.callback_block==0:
                arr=self.prearr2arr(x)
                print(arr[:1+self.depth])
                if self.callback_stop(self.callback_count,arr):
                    raise Exception('callback_stop condition satisfied... terminating minimize')
                if self.callback_count%(16*self.callback_block)==0:    
                    print(self.fu(x))
                    #print(x.tolist())
                    print(arr)

    def test_der(self,eps=2.^-13):
        arr=self.rand((10,25),(0.5,0.9))
        fu,der=self.fu_arr(arr),self.der_arr(arr)
        eps_precise=self.RR(eps) if self.precise else eps
        for j in range(self.total_length):
            arr_new=arr.copy()
            arr_new[j]+=eps
            fu_new=self.fu_arr(arr_new)
            compare((fu_new-fu)/eps_precise,der[j],'RSB bound derivative at coord'+str(j))
        print()
    
    def clock(self,iters=1):
        arrs=[self.rand((10,25),(0.5,0.9)) for _ in range(iters)]
        t0=time.time()
        for arr in arrs:
            self.fu_arr(arr)
        t1=time.time()
        for arr in arrs:
            self.der_arr(arr)
        t2=time.time()
        return (t1-t0)/iters,(t2-t1)/iters

class RSBoptimizer:

    def __init__(self,rsb):
        self.rsb=rsb
        self.funjac_kwargs={'fun':self.rsb.fu,'jac':self.rsb.der}
        #DUAL LOCOPT: self.basinhop_loc_kwargs={'method':'BFGS','options':{'gtol':1e-9,'maxiter':100}}
        self.set_loc_opt()
        self.default_setup()
        
        #dictionary storing certain arrays; possible keys: 'best','last','dist_lvl3'...
        self.storedx={}
        
    def default_setup(self):
        self.stepsize=0.25
        self.step_factor=0.75
        self.beta=1e7
        self.adaptive_beta_const=None#5.
        self.accept_rate=None
        self.lim_total=np.inf
        self.lim_cycle=np.inf
        self.step_adjust_block=32
        self.target=0.
        self.accept_info=False
        self.reject_info=False
        self.reset()
    
    def set_loc_opt(self,method='BFGS'):
    #for basinhop use smaller maxiter?
        if method=='BFGS':
            self.loc_opt_kwargs={'method':'BFGS','options':{'gtol':1e-10,'maxiter':10000,'disp':False}}  #starting from m_bounds=(0.99,0.99)
        elif method=='CG':
            self.loc_opt_kwargs={'method':'CG','options':{'gtol':1e-10,'maxiter':10000,'disp':False}} #CG options should be tested!!!
        elif method=='Nelder-Mead':
            self.loc_opt_kwargs={'method':'Nelder-Mead','tol':1e-10,'options':{'maxiter':500000,'adaptive':True,'disp':False}}
            #start=rsb.rand((17+deg/2,20+deg/2),(0.5,0.6),True)
        
    def reset(self):
        self.bestbnd=np.inf
        self.bestx=None
        self.cnt_total=0
        self.cnt_cycle=0
        self.cnt_since_best=0
        self.cnt_accept=0
        self.diffs=[]

    def param2str(self):
        return 'STEPSIZE:'+str(round(self.stepsize,6))+' BETA:'+str(int(round(self.beta)))        

    def store(self,name,prearr):
        self.storedx[name]=(prearr,self.rsb.prearr2arr(prearr))

    def loc_opt(self,start):
        return optimize.minimize(x0=start,**self.funjac_kwargs,**self.loc_opt_kwargs)
        #callback=self.rsb.callback

    def run_loc_opt(self,disp=True,bnd_limit=-np.inf):
        bestbnd=np.inf
        while bestbnd>bnd_limit:
            start=self.rsb.rand(prearr_output=True)
            res=self.loc_opt(start)
            if res.fun<bestbnd:
                bestbnd=res.fun
                self.store('best_loc_opt',res.x)                
                if disp:
                    print(bestbnd,self.loc_opt_kwargs)
                    print(res.message)
                    bestarr=rsb.prearr2arr(res.x)
                    self.rsb.print(bestarr,upto=0)
                    #print('vector sums diff:',self.rsb.vector_sums(bestarr))
                    print(rsb.fu_arr_precise(bestarr,300),'(precise value after normalize)')
                    #print("STARTED AT:")
                    #self.rsb.print(start,{'la','m'})
                    print()                
            else:
                if disp:
                    print(res.fun,res.message)
                    
        return res

    # should be re-written in a way that parameter-intervals are given
    # and after niters steps since best, we restart with newly sampled 
    # random parameters and with reverting back to bestx
    # but keeping the current step
    
    #should probably use much smaller stepsize when close to optimum...
    #smaller stepsize and larger beta
    #adaptive beta: 1/beta ~ distance to optimum
    #then constant accept_rate should automatically set ideal stepsize
    
    def check_if_best(self,disp=False,store=False):
        if self.bnd<self.bestbnd:
            self.bestbnd=self.bnd
            self.bestx=self.x.copy()
            if disp:
                print('COUNT:',self.cnt_cycle,self.cnt_total,' ',self.param2str())
                print(self.bestbnd,'(value, may be distorted)')
                self.rsb.print(self.bestx,upto=1,prearr_input=True) #upto=0
            if store:
                self.store('best',self.bestx)
            return True
        else:
            return False
    
    def basinhop(self,start=[],loc_opt_needed=False):        
        print("BASINHOPPING")
        self.rsb.print_parameters()
        print()
        if len(start)==0:
            start=self.rsb.rand(prearr_output=True)
        print("starting at:")
        self.rsb.print(start,prearr_input=True)

        self.x=self.loc_opt(start).x if loc_opt_needed else start.copy()
        self.bnd=self.rsb.fu(self.x)
        print(self.bnd,'(value)')
        self.check_if_best(disp=False)
        print()
        

        trueval=1.
        while True:
            self.cnt_total+=1
            self.cnt_cycle+=1

            self.bh_step()
                    
            if self.cnt_total>self.lim_total:
                break
            #if self.cnt_cycle>self.lim_cycle:
                #restart from self.storedx['best'] with possibly perturbed params

            #if self.cnt_total% self.beta_update_frequency==0:
            if self.adaptive_beta_const!=None:
                self.beta=max(1e3,self.adaptive_beta_const/median(self.diffs))
            #print('BETA:',self.beta)
                
            if self.accept_rate!=None and self.cnt_total%self.step_adjust_block==0:
                if self.accept_rate*self.step_adjust_block>self.cnt_accept:
                    self.stepsize*=self.step_factor
                else:
                    self.stepsize/=self.step_factor
                print('ACCEPTED:',str(self.cnt_accept)+'/'+str(self.step_adjust_block))    
                print(self.param2str())
                print(self.bnd,"(current bound)")
                self.rsb.print(self.x,{'m'},prearr_input=True)
                print()
                self.cnt_accept=0

        return (self.bestbnd,self.bestx)
    
    def bh_step(self):
        x_pert=self.rsb.perturbe(self.x,self.stepsize)
        res=self.loc_opt(x_pert)
        #DUAL LOCOPT: res=optimize.minimize(x0=x_pert,**self.funjac_kwargs,**self.basinhop_loc_kwargs)
        bnd_diff=res.fun-self.bnd
        step_strength=exp(min(0,-self.beta*bnd_diff))
        cut=random()
        percentage=str(round(100*step_strength,1))+'%'
        
        #update "step statistics"
        statsize=21
        self.diffs.append(abs(bnd_diff))
        if len(self.diffs)>statsize:
            self.diffs.pop(0)
            


        if step_strength>cut:                    
            self.cnt_accept+=1
            
            #DUAL LOCOPT: res=self.loc_opt(res.x)

            self.x=res.x
            self.bnd=res.fun

            if self.accept_info:
                print(self.bnd,percentage,' ',self.param2str())               
            if self.check_if_best(disp=True,store=True):  
                self.cnt_cycle=0                
            
                chk=self.rsb.prearr2arr(self.bestx)
                trueval=self.rsb.fu_arr_precise(chk,300)
                print(trueval,'(true value)')
                print()
                
                #target-adaptive beta
                #if self.target>0 and trueval>self.target:
                #    self.beta=1/(trueval-self.target)
                #    print("NEW BETA:",self.beta)
                
                lvl=0
                #lvl=self.rsb.distortion_level
                if lvl>0:
                    while trueval<self.rsb.distortion_thresholds[self.rsb.depth-lvl]:
                        lvl-=1
                        print("DISTORTION LEVEL CHANGED TO:",lvl,"at trueval",trueval)
                    if lvl<self.rsb.distortion_level:
                        self.rsb.distortion_level=lvl
                        self.storedx['lvl'+str(lvl)]=self.bestx
                        res=self.loc_opt(self.bestx)
                        self.x=res.x
                        self.bnd=res.fun
                        self.bestx=res.x.copy()
                        self.bestbnd=res.fun
                        
        elif self.reject_info:
            print('rejected',percentage,res.fun)

#RS=replica symmetric bound
class RSbound:
    ca=1

    def __init__(self,deg,nrs):        
        self.deg=deg
        self.nrs=nrs
        assert len(nrs)==1
        self.nr=nrs[0]
    
    #cf: factor constant (star: la ;edge: -1)
    def func(self,dim,cf,ms,prs):
        axes=list(range(dim))

        pr=1-prs[1] if dim==2 else prs[1]
        mat=np.full(tuple([self.nr]*dim),cf)
        mat=self.ca+mul(mat,pr,axes)

        mat=log(mat)
        mat=mul(mat,prs[0],axes)
        mat=block_sum(mat,self.nr,axes)

        return mat.flatten()[0]
    
    def bound(self,la,ms,prs):
        star=self.func(self.deg,la,ms,prs)
        edge=self.func(2,-1,ms,prs)
        return (star-self.deg*edge/2)/(log(la)) 

     
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

    
    #dim:  deg for star; 2 for edge
    def func_and_der(self,dim,cf,ms,prs):
        axes=list(range(dim))

        faux=[None]*(self.depth+1)
        daux=[None]*self.depth

        pr=1-prs[-1] if dim==2 else prs[-1]
        mat=np.full(tuple([self.nr_prod]*dim),cf)
        mat=self.ca+mul(mat,pr,axes)
        faux[-1]=mat

        for i in range(self.depth-1,-1,-1):
            daux[i]=ms[i]*mat^(ms[i]-1)
            daux[i]=mul(daux[i],prs[i],axes)
            mat=mat^ms[i]
            mat=mul(mat,prs[i],axes)
            mat=block_sum(mat,self.nrs[i],axes)
            faux[i]=mat
        nrm=faux[0].flatten()[0]    


        def der_cf():
            #mat=np.ones(tuple([self.nr_prod]*dim))
            #mat=mul(mat,pr,axes)
            mat=1
            for ax in axes:
                li=[1]*dim
                li[ax]=self.nr_prod
                sh=tuple(li)
                mat=mat*pr.reshape(sh)

            for i in range(self.depth-1,-1,-1):
                mat=mat*daux[i]
                mat=block_sum(mat,self.nrs[i],axes)            

            return mat.flatten()[0]/nrm

        def der_m():
            return [der_mk(k) for k in range(self.depth)]

        def der_mk(k):
            assert k<self.depth

            mat=faux[k+1]
            
            #could be inside for-loop...
            mat=log(mat)*mat^ms[k]
            mat=mul(mat,prs[k],axes)
            mat=block_sum(mat,self.nrs[k],axes)            
            
            for i in range(k-1,-1,-1):
                mat=mat*daux[i]
                mat=block_sum(mat,self.nrs[i],axes)            

            return mat.flatten()[0]/nrm
        
        def der_pr():
            return [der_prk(k) for k in range(self.depth+1)]

        def der_prk(k):
            assert k<self.depth+1

            mat=None
            if k<self.depth:
                mat=faux[k+1]
                mat=mat^ms[k]
                mat=mul(mat,prs[k],axes[1:])
                mat=block_sum(mat,self.nrs[k],axes[1:])
            else:
                mat=np.full(tuple([self.nr_prod]*dim),cf)
                pr=prs[k]
                if dim==2:
                    mat=-mat
                    pr=1-pr
                mat=mul(mat,pr,axes[1:])
            for i in range(k-1,-1,-1):
                factor=np.zeros_like(mat)
                fill=daux[i]
                block=int(factor.shape[0]/fill.shape[0])
                factor=np.repeat(fill,block,axis=0)

                mat=mat*factor
                mat=block_sum(mat,self.nrs[i],axes[1:])
            
            res=(dim/nrm)*mat.flatten()
            return res
                    
        der=np.concatenate([np.array([der_cf()]),der_m()]+der_pr())
        return (log(nrm),der)

    def bound(self,la,ms,prs):
        star=self.func(self.deg,la-1,ms,prs)
        edge=self.func(2,1/la-1,ms,prs)
        m_prod=np.prod(ms)
        return (star-self.deg*edge/2)/(m_prod*log(la)) 

    def bound_der(self,la,ms,prs):
        star,star_der=self.func_and_der(self.deg,la-1,ms,prs)
        edge,edge_der=self.func_and_der(2,1/la-1,ms,prs)
        edge_der[0]*=(-1/la**2)
        derarr=star_der-self.deg*edge_der/2
        m_prod=np.prod(ms)
        derarr=derarr/(m_prod*log(la))
        fun=(star-self.deg*edge/2)/(m_prod*log(la))
        secondterm=np.zeros_like(derarr)
        secondterm[0]=-fun/(la*log(la))
        for i in range(len(ms)):
            secondterm[i+1]=-fun/ms[i]        

        return derarr+secondterm
                
class TwoRSBbound:
    
    def __init__(self,deg,nrs):
        assert len(nrs)==1
        self.num=nrs[0]
        print('initializing TwoRSBbound('+str(deg)+','+str(self.num)+')...')
        #self.nrs=nrs
        self.deg=deg
        prtns=partitions(deg,self.num)
        self.parr=np.array(prtns,int)
        self.coeff=np.array([multinomial(*part) for part in prtns],int)
        prtns_der=partitions(deg-1,self.num)
        self.parr_der=np.array(prtns_der,int)
        self.coeff_der=np.array([multinomial(*part) for part in prtns_der],int)
        print('initialization finished.')
    
    def bound(self,la,ms,prs):
        m,p0,p1=ms[0],prs[0],prs[1]    
        p0_powers=powers(p0,self.deg)
        p1_powers=powers(p1,self.deg)
                
        def product(powers,part):
            prod=1
            for i in range(self.num):
                prod*=powers[i,part[i]]
            return prod
            
        prod0=np.array([product(p0_powers,part) for part in self.parr])
        prod1=np.array([product(p1_powers,part) for part in self.parr])
        star=np.dot(self.coeff,prod0*(1+(la-1)*prod1)**m)
        
        p1f=1-p1
        mat=(1+(1/la-1)*p1f.reshape((1,-1))*p1f.reshape((-1,1)))^m
        edge=np.dot(np.dot(mat,p0),p0)
                
        return (log(star)-self.deg/2*log(edge))/(m*log(la))

    def bound_der(self,la,ms,prs):
        m,p0,p1=ms[0],prs[0],prs[1]    
        p0_powers=powers(p0,self.deg)
        p1_powers=powers(p1,self.deg)
                
        def product(powers,part):
            prod=1
            for i in range(self.num):
                prod*=powers[i,part[i]]
            return prod
            
        prod0=np.array([product(p0_powers,part) for part in self.parr])
        prod1=np.array([product(p1_powers,part) for part in self.parr])
        prod0_dp0=np.array([product(p0_powers,part) for part in self.parr_der])
        prod1_dp0=np.array([[product(p1_powers,part)*p1[j] for part in self.parr_der] for j in range(self.num)])
        prod1_dp1=(la-1)*np.array([[0 if part[j]==0 else part[j]*product(p1_powers,part)/p1[j] for part in self.parr] for j in range(self.num)])        
        
        star_dp0=np.dot(prod0_dp0*(1+(la-1)*prod1_dp0)**m,self.coeff_der)
        star=np.dot(star_dp0,p0)

        arr=1+(la-1)*prod1
        arr_dm=log(arr)*arr**m
        star_dm=np.dot(self.coeff,prod0*arr_dm)

        aux=m*arr**(m-1)
        star_dla=np.dot(self.coeff,prod0*prod1*aux)
        
        arr_dp1=prod0*prod1_dp1*aux
        star_dp1=np.dot(arr_dp1,self.coeff)
        
        p1f=1-p1
        premat=p1f.reshape((1,-1))*p1f.reshape((-1,1))
        mat=1+(1/la-1)*premat
        mat_dla=(-m/la^2)*premat*mat**(m-1)
        mat_pow=mat**m
        edge_dp0=np.dot(mat_pow,p0)
        edge=np.dot(edge_dp0,p0)
        mat_dm=log(mat)*mat_pow
        edge_dm=np.dot(np.dot(mat_dm,p0),p0)
        edge_dla=np.dot(np.dot(mat_dla,p0),p0)
        edge_dp1=np.dot( (-m*(1/la-1))*mat**(m-1)*p1f.reshape((1,-1)), p0)*p0
        
        dla=(star_dla/star-self.deg/2*edge_dla/edge)/(m*log(la))-(log(star)-self.deg/2*log(edge))/(m*la*log(la)^2)
        dm=(star_dm/star-self.deg/2*edge_dm/edge)/(m*log(la))-(log(star)-self.deg/2*log(edge))/(m^2*log(la))
        dp0=self.deg*(star_dp0/star-edge_dp0/edge)/(m*log(la))
        dp1=(star_dp1/star-self.deg*edge_dp1/edge)/(m*log(la))
        #MAKE dp0 CENTERED???
        #dp0-=(dp0.sum()/self.num)
        return np.concatenate([np.array([dla,dm]),dp0,dp1])                

class ThreeRSBbound:
    
    def __init__(self,deg,nrs):
        assert len(nrs)==2
        print('initializing ThreeRSBbound('+str(deg)+',',nrs,')...')
        self.nrs=nrs
        #self.num=nrs[0]*nrs[1]
        self.deg=deg
        outer_prtns=partitions(deg,nrs[0])
        self.outer_parr=np.array(outer_prtns,int)
        self.outer_coeff=[multinomial(*outer_part) for outer_part in outer_prtns]
        self.inner_coeff=[]
        self.inner_parr=[]
        for outer_part in outer_prtns:
            prtns=[]
            for i in range(nrs[0]):
                prtns.append(partitions(outer_part[i],nrs[1]))
            ii=[0]*nrs[0]
            im=[len(prtns[i]) for i in range(nrs[0])]
            lili=[]
            lili_coeff=[]
            while True:
                li=[]
                prod=1
                for i in range(nrs[0]):
                    pt=prtns[i][ii[i]]
                    li+=pt
                    prod*=multinomial(*pt)

                lili.append(li)    
                lili_coeff.append(prod)
                
                j=1
                while j<=nrs[0]:
                    ii[-j]+=1
                    if ii[-j]==im[-j]:
                        ii[-j]=0
                        j+=1
                    else:
                        break
                if j>nrs[0]:
                    break
            
            self.inner_parr.append(np.array(lili,int))
            self.inner_coeff.append(np.array(lili_coeff,int))
        print('initialization finished.')
    
    def bound(self,la,ms,prs):
        p0,p1,p2=prs[0],prs[1],prs[2]
        p0_powers=powers(p0,self.deg)
        p1_powers=powers(p1,self.deg)
        p2_powers=powers(p2,self.deg)
                
        def product(powers,part):
            prod=1
            for i in range(len(part)):
                prod*=powers[i,part[i]]
            return prod

        star=0
        for i in range(len(self.outer_coeff)):
            parr=self.inner_parr[i]
            coeff=self.inner_coeff[i]
            prod1=np.array([product(p1_powers,part) for part in parr])
            prod2=np.array([product(p2_powers,part) for part in parr])
            val=np.dot(coeff,prod1*(1+(la-1)*prod2)^ms[1])
            prod0=product(p0_powers,self.outer_parr[i])
            star+=self.outer_coeff[i]*prod0*val^ms[0]
        
        p2f=1-p2
        mat=(1+(1/la-1)*p2f.reshape((1,-1))*p2f.reshape((-1,1)))^ms[1]
        mat=mul(mat,p1,[0,1])
        mat=block_sum(mat,self.nrs[1],[0,1])^ms[0]
        edge=np.dot(np.dot(mat,p0),p0)
                
        return (log(star)-self.deg/2*log(edge))/(ms[0]*ms[1]*log(la))
             
def test_3RSB():
    deg=3
    nrs=[3,3]
    rsb=RSBwrapper(deg,nrs,300,False)
    la=23.1
    ms=np.array([0.59,0.73])
    r1=np.array([0.47           ,0.23                   ,0.3                ])
    r2=np.array([0.38,0.36 ,0.26,0.07 ,0.57    ,0.36    ,0.1  ,0.6    ,0.3  ])
    r3=np.array([0.02,0.196,0.79,0.525,0.002444,0.064427,0.051,0.98442,0.482])
    val=rsb.comp.bound(la,ms,[r1,r2,r3])
    arr=np.concatenate([np.array([la]),ms,r1,r2,r3])
    print(rsb.vector_sums(arr))
    val_precise=rsb.fu_arr(arr)
    compare(val,val_precise,"3RSB bound")
    print(rsb.fu_arr_precise(arr,300))
    print(rsb.fu_arr_precise(arr,500))
    print(rsb.fu_arr_precise(arr,200))
    print('0.4507881549028990413784822935938187148')
    print()

#this functionality should be a built-in feature of the RSBwrapper class
#class FixedCoord:
#    def __init__(self,orig,fix):

def test_TwoRSB():
    rsb=RSBwrapper(18,[4],TwoRSBbound)
    #0.20680937348995632
    la=[35.09951093785]
    m=[0.9123077698666]
    p0=[0.21144997,0.01516656 ,0.00783117,0.76555230]
    p1=[0.0005961 ,0.055952454,0.37966684,0.99895792]
    arr=np.array(la+m+p0+p1)
    print(rsb.fu_arr(arr))
    print(rsb.fu_arr_precise(arr,300))
    print('0.20680937348995')
    print(rsb.der_arr(arr).tolist())
    print()

def test_ThreeRSB():
    #0.3119222269268 
    comp=ThreeRSBbound(8,[3,2])
    comp2=RSBbound(8,[3,2])
    la=25.47142905
    ms=np.array([0.85091021,0.84740723])
    p0=np.array([0.51474384,0.3573083,0.12794786])
    p1=np.array([0.83535003,0.16464997,0.4709506,0.5290494,0.40594239,0.5940576])
    p2=np.array([0.999999999,0.3453801,0.82471222,0.04716774,0.24350653,0.00116882])
    prs=[p0,p1,p2]
    print(comp.bound(la,ms,prs))
    print(comp2.bound(la,ms,prs))

def compare_TwoRSB(deg,nrs):
    rsbs=[]
    rsbs.append(RSBwrapper(deg,nrs,RSBbound,300,False))
    rsbs.append(RSBwrapper(deg,nrs,TwoRSBbound,300,False))
    rsbs.append(RSBwrapper(deg,nrs,RSBbound))
    rsbs.append(RSBwrapper(deg,nrs,TwoRSBbound))
    arr=rsbs[0].rand((15,20),(0.5,0.9))
    for rsb in rsbs:
        print(rsb.fu_arr(arr))
    res=[rsb.der_arr(arr) for rsb in rsbs]
    for i in range(len(res[0])):
        for re in res:
            print(re[i])
    print()        

def run_deg3_rsb2(num):
    rsb=RSBwrapper(3,[num])

    rsb.m_distortion=True
    rsb.m_distortion_limit=0.9
    rsb.m_distortion_constant=2.^-13
    rsb.perturbation_mode='prlast'
    rsb.print_parameters()

    opt=RSBoptimizer(rsb)
    opt.stepsize=0.125
    opt.beta=5e10
    opt.accept_info=True

    opt.basinhop(loc_opt_needed=True)

def random_nrs(deg,max_pr):
    max_depth=int(floor(log(max_pr)/log(2)))
    if deg>7:
        max_depth=1
    depth=rndm.randint(1,max_depth)
    nrs=[0]*depth
    for i in range(depth):
        max_nr=int(floor(max_pr/2^(depth-i-1)))
        nrs[i]=rndm.randint(2,max_nr)
        max_pr=int(floor(max_pr/nrs[i]))
    return nrs

def run_opt(rsb,method='BFGS',start=np.empty((0,))):
    print(rsb.deg,rsb.nrs)
    if method=='BFGS': 
        kwargs={'fun':rsb.fu,'jac':rsb.der,'method':'BFGS','callback':rsb.callback,'options':{'gtol':1e-10,'maxiter':10000,'disp':False}}
    elif method=='NM':
        kwargs={'fun':rsb.fu,'method':'Nelder-Mead','tol':1e-10,'callback':rsb.callback,'options':{'maxiter':500000,'adaptive':True,'disp':False}}
    else:
        assert False
    if len(start)==0:
        start=rsb.rand()
    prestart=rsb.arr2prearr(start)
    #print(start.tolist())
    res=optimize.minimize(x0=prestart,**kwargs)
    #print(res.fun)
    arr=rsb.prearr2arr(res.x)
    print(arr[:1+len(rsb.nrs)].tolist())
    prectrueval=rsb.fu_arr_precise(arr,300)
    print(float(prectrueval))
    
    result={}
    result['id']=rndm.randint(1,2^30)
    result['deg']=rsb.deg
    result['nrs']=rsb.nrs
    result['bound']=prectrueval
    result['method']=method
    result['arr']=arr
    result['prearr']=res.x
    result['start']=start
    result['prestart']=prestart
    result['rsb_setup']=(str(type(rsb.comp)),rsb.offset_constant,rsb.m_distortion,rsb.m_distortion_limit,rsb.m_distortion_constant)
    result['msg']=res.message

    return result

def save_result(res):
    li=load('saved_results.sobj')
    li.append(res)
    save(li,'saved_results')

def find_result(id_nr):
    li=load('saved_results.sobj')
    for res in li:
        if res['id']==id_nr:
            return res
    return None

def save_if_top(res):
    top_nr=20
    bnd=res['bound']
    deg=res['deg']
    nrs=res['nrs']
    depth=len(nrs)
    line=(float(bnd),nrs,res['id'],res['method'])

    top_all=load('topscores.sobj')
    top=top_all[deg][depth]
    added=False
    for i in range(len(top)):
        if bnd<top[i][0]:
            top.insert(i,line)
            added=True
            print('#'+str(i+1),'in',deg,depth)
            break
    if added==False and len(top)<top_nr:
        top.append(line)
        print('last:#'+str(len(top)),'in',deg,depth)
        added=True
    if len(top)>top_nr:
        top.pop(-1)
    print()
    save(top_all,'topscores')
    if added:
        save_result(res)

def ts_print(deg=0):
    li=load('topscores.sobj')
    def prnt(d):
        tsd=li[d]
        print('deg=',d)
        for ts in tsd:
            for s in ts:
                print(float(s[0]),s[1],s[2],s[3])
            print()

    if deg<3:
        for d in range(3,20):
            prnt(d)
    else:
        prnt(deg)

def run_forever(deg,nrs):
    if deg<3:
        deg=rndm.randint(3,19)
    good_starting_la=[]    
    while True:
        print('good la to start with:',good_starting_la)
        rsb=RSBwrapper(deg,nrs,TwoRSBbound)
        #rsb=RSBwrapper(deg,nrs,ThreeRSBbound)
        rsb.m_distortion=True
        #rsb.m_distortion_limit=0.9#0.7
        #rsb.m_distortion_constant=2.^-13#15
        rsb.m_distortion_limit=0.98
        rsb.m_distortion_constant=1.

        rsb.la_bounds=(15.,50.)
        if deg>11:
            rsb.la_bounds=(40.,80.)
        if deg>15:
            rsb.la_bounds=(80.,110.) #for 17,[4]

        rsb.callback_enabled=True
        rsb.callback_block=8
        bad_la_dict={10:18.8,11:19.5,12:20.3,13:21.0,14:21.96,15:22.91,16:23.85,17:24.79,18:25.71,19:26.63}
        bad_la=bad_la_dict[deg]
        #rsb.callback_stop=lambda iters,arr: iters>2*16*8 and arr[0]<bad_la+1
        rsb.callback_stop=lambda iters,arr: arr[0]<bad_la+0.125

        try:
            res=run_opt(rsb,method='BFGS')
        except Exception as err:
            print()
            print(type(err))
            print(err.args)
            print(err)
            print()
            continue

        #for ThreeRSBbound
        #rsb.callback_block=8
        #res=run_opt(rsb,method='NM')

        starting_la=round(res['start'][0],1)
        good_starting_la.append(starting_la)
        save_if_top(res)

np.set_printoptions(suppress=True)
#np.set_printoptions(precision=4)

#run_forever(13,[4])

#rsb=RSBwrapper(14,[5],TwoRSBbound)
#rsb.m_distortion=True
#opt=RSBoptimizer(rsb)
#opt.accept_info=True
#opt.reject_info=True
#start=np.array([31.,0.99,0.4,0.08,0.04,0.08,0.4,0.01,0.1,0.5,0.8,0.99])
#prestart=rsb.arr2prearr(start)
#opt.basinhop(prestart,loc_opt_needed=True)

#rsb=RSBwrapper(15,[4],TwoRSBbound)
#rsb.m_distortion=True
#opt=RSBoptimizer(rsb)
#opt.accept_info=True
#opt.reject_info=True
#instead of random start, use something like this: #start=np.array([31.,0.99,0.4,0.1,0.1,0.4,0.01,0.1,0.8,0.99])
#prestart=rsb.arr2prearr(start)
#opt.basinhop(loc_opt_needed=True)

#test_3RSB()
#test_TwoRSB()
#test_ThreeRSB()
#compare_TwoRSB(4,[5])
#RSBwrapper(10,[4],TwoRSBbound).test_der()
#RSBwrapper(10,[4],TwoRSBbound,300,False).test_der(2.^-45)
#RSBwrapper(3,[4,2],300,False).test_der(2.^-45)
#RSBwrapper(4,[5,3],300,False).test_der(2.^-45)

#for deg in range(7,20):
#    for num in range(2,7):
#        #print(deg,num)
#        rsb=RSBwrapper(deg,[num],TwoRSBbound)
#        print(rsb.clock())


#rsb=RSBwrapper(4,[4],TwoRSBbound,300,True)
#rsb=RSBwrapper(4,[4],TwoRSBbound)
#rsb=RSBwrapper(3,[4,2],300,True)
#rsb=RSBwrapper(3,[5,3,2])

#rsb.m_distortion=True
#rsb.m_distortion_limit=0.9#0.7
#rsb.m_distortion_constant=2.^-13#15
#rsb.perturbation_mode='prlast'
#rsb.perturbation_mode='default'
#rsb.rel_step['la']=0.
#rsb.rel_step['m']=0.
#rsb.print_parameters()

#opt=RSBoptimizer(rsb)
#opt.stepsize=0.25
#opt.beta=1e8
#opt.accept_info=True
#opt.reject_info=True

#opt.run_loc_opt()
#opt.basinhop(loc_opt_needed=True)

#works but stepsize is uniform for all coordinates here
#so at least prearr[0] (pre-lambda) should be changed...
#start=rsb.rand(prearr_output=True)
#target=0.450786
#res=optimize.basinhopping(lambda x:rsb.fu(x)-target,start,niter=1000,stepsize=0.1,T=1e-9,minimizer_kwargs={'method':'BFGS','jac':rsb.der,'options':{'gtol':1e-10,'maxiter':200}},disp=True)
#print(res)                            

#rsb=RSBwrapper(3,[1],RSbound)
#first_moment=rsb.comp.bound(244.7372427164263,np.array([1.]),[np.array([1.]),np.array([1-0.8486410879780224])])
#print(first_moment)
#rsb=RSBwrapper(3,[2],RSbound)
#run_opt(rsb,'NM')