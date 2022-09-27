import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy import sparse,linalg
from scipy.sparse import linalg as sparse_linalg



twist_angle=lambda p,q: np.arccos((3*p**2+3*p*q+q**2/2)/(3*p**2+3*p*q+q**2))
number_of_sublattice_positions=lambda p,q: p**2+p*q+q**2//3 if q%3==0 else 3*p**2+3*p*q+q**2
rot=lambda theta: np.array([[np.cos(theta),-np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,1.0]])



def superlattice(p,q):
    a=2.46 #Angstrom
    s3=np.sqrt(3)
    a1=a*np.array([s3/2,-1/2,0.0])
    a2=a*np.array([s3/2,1/2,0.0])
    def positions(i11,i12,i21,i22):
        A=[]
        D=i11*i22-i21*i12
        ms=(0,i11,i21,i11+i21)
        ns=(0,i12,i22,i12+i22)
        for m in range(min(ms),max(ms)+1):
            for n in range(min(ns),max(ns)+1):
                mm=(m*i22-i21*n)/D
                nn=(i11*n-m*i12)/D
                if 0<=mm<1 and 0<=nn<1:
                    A.append(m*a1+n*a2)
        B=[p+(a1+a2)/3 for p in A]
        return A+B
    if q%3==0:
        i11=p+q//3
        i12=q//3
        i21=-q//3
        i22=p+2*q//3
    else:
        i11=p
        i12=p+q
        i21=-p-q
        i22=2*p+q
    L1=i11*a1+i12*a2
    L2=i21*a1+i22*a2
    r1=positions(i11,i12,i21,i22)
    Mac=np.array([[a1[0],a2[0],0.0],[a1[1],a2[1],0.0],[0.0,0.0,1.0]])
    Mca=np.linalg.inv(Mac)
    i11r,i12r,_=np.round(Mca@rot(-twist_angle(p,q))@Mac@[i11,i12,0]).astype('int')
    i21r,i22r,_=np.round(Mca@rot(-twist_angle(p,q))@Mac@[i21,i22,0]).astype('int')
    r2=positions(i11r,i12r,i21r,i22r)@rot(twist_angle(p,q)).T + np.array([0.0,0.0,3.35])
    return np.concatenate((r1,r2)),L1,L2



@jit(nopython=True)
def hopping_max_index(L1,L2,max_distance):
    M=-1 #maximum index of iteration
    min_dist=-1.
    while min_dist<max_distance:
        M+=1
        min_dist=np.inf
        for O in [-M*(L1+L2),-M*L1+(M+1)*L2,(M+1)*L1-M*L2,(M+1)*(L1+L2)]:
            for v in [L1,L2]:
                for P in [0*L1,L1,L2,L1+L2]:
                    parameter=np.dot(v,P-O)/np.dot(v,v)
                    dist=np.linalg.norm(O+parameter*v-P)
                    if min_dist>dist:
                        min_dist=dist
    return M



def hoppings(r,L1,L2,nhops,max_distance):
    """Naive hopping finder. Use this for very small unit cells (pragmatically, only the pure AA and AB cases)."""
    hops=np.zeros((nhops,8))
    ix=0 #keep track of the number of hoppings
    M=hopping_max_index(L1,L2,max_distance)
    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            for m in range(-M,M+1):
                for n in range(-M,M+1):
                    if np.linalg.norm(r[j,:]-r[i,:]+m*L1+n*L2)<max_distance and not(i==j and m==n==0): #remove on-site hoppings
                        hops[ix,0] = i
                        hops[ix,1] = j
                        hops[ix,2:5] = r[i,:]
                        hops[ix,5:8] = r[j,:] + m*L1+n*L2
                        ix+=1
    return hops[0:ix]



@jit(nopython=True)
def hoppings_TBG(r,L1,L2,nhops,max_distance):
    """More efficient hoppings finder, only look for i<j hoppings."""
    hops=np.zeros((nhops,8))
    ix=0 #keep track of the number of hoppings
    M=hopping_max_index(L1,L2,max_distance)
    for i in range(r.shape[0]):
        for j in range(i+1,r.shape[0]):
            for m in range(-M,M+1):
                for n in range(-M,M+1):
                    if np.linalg.norm(r[j,:]-r[i,:]+m*L1+n*L2)<max_distance:
                        hops[ix,0] = i
                        hops[ix,1] = j
                        hops[ix,2:5] = r[i,:]
                        hops[ix,5:8] = r[j,:] + m*L1+n*L2
                        ix+=1
    return hops[0:ix]



def t_intra(x1,y1,x2,y2):
    a=2.46 #Angstrom
    x=x2-x1
    y=y2-y1
    r=np.sqrt(x**2+y**2)/a
    t=np.zeros_like(r)
    t[(0.3  < r) * (r < 0.8 )] = -2.8922   #1st neighbors = 0.5774
#     t[(0.8  < r) * (r < 1.1 )] =  0.2425  #2nd neighbors = 1.0
#     t[(1.1  < r) * (r < 1.3 )] = -0.2656  #3rd neighbors = 1.1547
#     t[(1.3  < r) * (r < 1.6 )] =  0.0235  #4th neighbors = 1.5275
#     t[(1.6  < r) * (r < 1.8 )] =  0.0524  #5th neighbors = 1.7321
#     t[(1.8  < r) * (r < 2.05)] = -0.0209  #6th neighbors = 2.0
#     t[(2.05 < r) * (r < 2.1 )] = -0.0148  #7th neighbors = 2.0817
#     t[(2.1  < r) * (r < 2.35)] = -0.0211  #8th neighbors = 2.3094
    return t



def t_inter(x1,y1,x2,y2,l1,l2,theta):
    a=2.46 #Angstrom
    x=x2-x1
    y=y2-y1
    r=np.sqrt(x**2+y**2)

    l0, l3, l6 = 0.3155, -0.0688, -0.0083
    xi0, xi3, xi6 = 1.7543, 3.4692, 2.8764
    x3, x6 = 0.5212, 1.5206
    k0, k6 = 2.0010, 1.5731

    rn=r/a
    V0=l0 * np.exp(-xi0 * rn**2) * np.cos(k0*rn)
    V3=l3 * rn**2 * np.exp(-xi3 * (rn-x3)**2)
    V6=l6 * np.exp(-xi6 * (rn-x6)**2) * np.sin(k6 * rn)

    c3 = lambda x: 4*x**3-3*x # cosseno do arco triplo
    c6 = lambda x: 32*x**6-48*x**4+18*x**2-1 # cosseno do arco sextuplo
    xx=np.nan_to_num(x/r) #cosseno direcional na direção x
    yy=np.nan_to_num(y/r) #cosseno direcional na direção y
    cos3=l1*c3(xx)-l2*c3(xx*np.cos(theta)+yy*np.sin(theta)) #termo chato que muda de sinal dependendo da sub-rede
    cos6=c6(xx)+c6(xx*np.cos(theta)+yy*np.sin(theta)) #termo par, não depende da sub-rede

    return V0+V3*cos3+V6*cos6



def hopping_parameters(hops,N,theta):
    intra=np.abs(hops[:,4]-hops[:,7])==0.0
    inter=np.invert(intra)
    l1=np.array([1 if i<N else -1 for i in hops[inter,0]])
    l2=np.array([1 if 2*N<=j<3*N else -1 for j in hops[inter,1]])
    t=np.zeros(hops.shape[0])
    t[intra]=t_intra(hops[intra,2],hops[intra,3],hops[intra,5],hops[intra,6])
    t[inter]=t_inter(hops[inter,2],hops[inter,3],hops[inter,5],hops[inter,6],l1,l2,theta)
    return t



def hamiltonian(hops,t,N,interlayer=0.0,V=0.0):
    i=hops[:,0].astype('int')
    j=hops[:,1].astype('int')
    r1=hops[:,2:5]
    r2=hops[:,5:8]
    tt=np.copy(t)
    inter=np.abs(hops[:,4]-hops[:,7])!=0.0
    tt[inter]=tt[inter]*interlayer
    Vdiag=V*np.concatenate((np.ones(2*N),-np.ones(2*N)))
    Vmat=sparse.diags(Vdiag)
    def H(k):
        data=tt*np.exp(1j*(r2-r1)@k)
        mat=sparse.coo_matrix((data, (i, j)),shape=(4*N,4*N))
        mat.eliminate_zeros()
        return mat+mat.getH()+Vmat
    return H



def eigenenergies(Hk):
    return linalg.eigh(Hk,eigvals_only=True)



def eigenenergies_sparse(Hk,nbands,Ef):
    return sparse_linalg.eigsh(Hk,k=nbands,sigma=Ef,return_eigenvectors=False)



def kticks(pts_per_line_segment):
    ticks=np.array([sum(pts_per_line_segment[0:i])%sum(pts_per_line_segment) for i in range(len(pts_per_line_segment)+1)])
    ticks[-1]+=sum(pts_per_line_segment)-1
    return ticks



def kpath(kpts,pts_per_line_segment):
    gamma=[] #path of kpoints in R² space (remember MAT-36)
    ell=[] #lengths of the length parameterized path (again, remember MAT-36)
    for n in range(len(kpts)-1):
        t=np.linspace(0,1,pts_per_line_segment[n],endpoint=False).reshape(-1,1)
        gamma_n=kpts[n]+t*(kpts[n+1]-kpts[n])
        ell_n=np.linalg.norm(gamma_n-kpts[n],axis=-1)
        if n>=1:
            ell_n+=ell[-1][-1]+(ell[-1][-1]-ell[-1][-2])
        gamma.append(gamma_n)
        ell.append(ell_n)
    gamma=np.concatenate(gamma)
    ell=np.concatenate(ell)
    return gamma,ell,kticks(pts_per_line_segment)



def bands(H,gamma):
    return np.array([eigenenergies(H(k).toarray()) for k in gamma])



def bands_sparse(H,gamma,nbands,Ef):
    return np.array([eigenenergies_sparse(H(k),nbands,Ef) for k in gamma])



class TwistedBilayerGraphene:
    def __init__(self,p,q):
        theta=twist_angle(p,q)
        N=number_of_sublattice_positions(p,q)
        r,L1,L2=superlattice(p,q)

        L3=np.array([0,0,1])
        G1=2*np.pi*np.cross(L2,L3)/np.dot(L1,np.cross(L2,L3))
        G2=2*np.pi*np.cross(L3,L1)/np.dot(L1,np.cross(L2,L3))

        #K[0],M[0],Kp[0],Mp[0],K[1],...
        kpt=lambda m,n: m*G1+n*G2
        Gamma=kpt(0,0)
        K=kpt(1/3,-1/3),kpt(1/3,2/3),kpt(-2/3,-1/3)
        Kp=kpt(2/3,1/3),kpt(-1/3,1/3),kpt(-1/3,-2/3)
        M=kpt(1/2,0),kpt(0,1/2),kpt(-1/2,-1/2)
        Mp=kpt(1/2,1/2),kpt(-1/2,0),kpt(0,-1/2)
        self.path_GMKG=[Gamma,M[0],K[0],Gamma]
        self.path_KGMKp=[K[0],Gamma,M[1],Kp[1]]
        
        if p==0 or q==0:
            self._hoppings=hoppings
            self._bands=lambda H,gamma,nbands,Ef: bands(H,gamma)
        else:
            self._hoppings=hoppings_TBG
            self._bands=bands_sparse
        
        self.theta=theta
        self.N=N
        self.L1=L1
        self.L2=L2
        self.r=r
        self.p=p
        self.q=q
        self.G1=G1
        self.G2=G2
        self.kpt=kpt
        self.Gamma=Gamma
        self.K=K
        self.Kp=Kp
        self.M=M
        self.Mp=Mp
        
    
    def calc_hops(self,max_distance):
        self.hops=self._hoppings(self.r,self.L1,self.L2,400*self.N,max_distance)
        self.t=hopping_parameters(self.hops,self.N,self.theta)

        
    def set_hamiltonian(self,interlayer=0.0,V=0.0):
        self.H=hamiltonian(self.hops,self.t,self.N,interlayer,V)
        

    def set_kpath(self,kpts,pts_per_line_segment):
        self.gamma,self.ell,self.kticks=kpath(kpts,pts_per_line_segment)

        
    def calc_bands(self,nbands,Ef):
        self.bands=self._bands(self.H,self.gamma,nbands,Ef)
        
