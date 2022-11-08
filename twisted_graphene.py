"""
HOME-MADE TIGHT-BINDING CODE FOR TWISTED BILAYER GRAPHENE

Contains all relevant constants, functions and classes.

Is meant to be imported as a module in other python scripts.
"""



import numpy as np #arrays
import matplotlib.pyplot as plt #plotting
import matplotlib #plot settings
from numba import jit #just-in-time compiled code (faster code).
from scipy import sparse #sparse matrices (efficient data storage)
from scipy import linalg #linear algebra routines for small matrices
from scipy import constants #convenient shortcut for physical constants (hbar, e, c ...)
from scipy.sparse import linalg as sparse_linalg #linear algebra for big sparse matrices



np.set_printoptions(linewidth=200,suppress=True) #increase "print" linewidth and suppress scientific notation


#plot settings
matplotlib.rc('text',usetex=True)
plt.rcParams['figure.figsize'] = (4.9,3.5)
plt.rcParams['font.size'] = 11.0
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Palatino'
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 200 #to preview plot in jupyter
plt.rcParams['figure.autolayout'] = True


#twist angle of commensurable structures, as a function of a pair of integers p,q
twist_angle=lambda p,q: np.arccos((3*p**2+3*p*q+q**2/2)/(3*p**2+3*p*q+q**2))

#number of positions inside unit cell per sublayer: N_\ell (notation I used in my TG)
number_of_sublattice_positions=lambda p,q: p**2+p*q+q**2//3 if q%3==0 else 3*p**2+3*p*q+q**2

#rotation matrix in xy plane (the plane of the material
rot=lambda theta: np.array([[np.cos(theta),-np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,1.0]])

#magnetic flux quantum (2.06e-15)
phi0=constants.h/(2*constants.e)



def superlattice(p,q):
    """
    Given the pair of integers p,q, return the correspondent crystal (basis r + primitive vectors L1,L2):
    
    r, L1, L2 = superlattice(p,q)
    """
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
    return np.concatenate((r1,r2)),L1,L2 #r,L1,L2



@jit(nopython=True)
def hopping_max_index(L1,L2,max_distance):
    """
    This function uses some analytic geometry to calculate the maximum index necessary for iteration
    in the hoppings function below, for a diamond shape ("losango regular") unit cell.
    """
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



@jit(nopython=True)
def hoppings(r,L1,L2,nhops,max_distance):
    """
    Only looks for i<j hoppings. Returns (:,8) array.
    
    hops = hoppings(r,L1,L2,nhops,max_distance)
    hops[:,0] #rows
    hops[:,1] #columns
    hops[:,2:5] #initial position vectors
    hops[:,5::] #final position vectors
    """
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



def hoppings_onsite(L1,L2,max_distance):
    """Generate hopping from a site to itself and its periodic repetitions. Returns (nhops,3) array."""
    M=hopping_max_index(L1,L2,max_distance)
    lat_vecs=[]
    for m in range(-M,M+1):
        for n in range(-M,M+1):
            R=m*L1+n*L2
            if np.linalg.norm(R)<max_distance:
                lat_vecs.append(R)
    return np.array(lat_vecs)



distances_intra=[x*2.46 for x in [0.01,0.58,1.01,1.16,1.53,1.74,2.01,2.09,2.31]]

def t_intra(x1,y1,x2,y2):
    """
    Intralayer hoppings of graphene.
    
    (x1,y1) and (x2,y2) are the initial and final positions, respectively.
    """
    a=2.46 #Angstrom. Lattice constant of graphene
    x=x2-x1
    y=y2-y1
    r=np.sqrt(x**2+y**2)/a
    t=np.zeros_like(r)
    t[(0.3  < r) * (r < 0.8 )] = -2.8922   #1st neighbors = 0.5774
    t[(0.8  < r) * (r < 1.1 )] =  0.2425  #2nd neighbors = 1.0
    t[(1.1  < r) * (r < 1.3 )] = -0.2656  #3rd neighbors = 1.1547
    t[(1.3  < r) * (r < 1.6 )] =  0.0235  #4th neighbors = 1.5275
    t[(1.6  < r) * (r < 1.8 )] =  0.0524  #5th neighbors = 1.7321
    t[(1.8  < r) * (r < 2.05)] = -0.0209  #6th neighbors = 2.0
    t[(2.05 < r) * (r < 2.1 )] = -0.0148  #7th neighbors = 2.0817
    t[(2.1  < r) * (r < 2.35)] = -0.0211  #8th neighbors = 2.3094
    return t



def t_inter(x1,y1,x2,y2,l1,l2,theta):
    """
    Interlayer hoppings of graphene.
    
    l1 and l2 are the sublayers associated to the initial (x1,y1) and final (x2,y2) positions, respectively.
    
    theta is the twist angle.
    """
    a=2.46 #Angstrom. Lattice constant of graphene
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



def hopping_parameters(hops,hops_onsite,N,theta,t_intra,t_inter):
    """
    Calculates off-site (t) and on-site (t_onsite) hopping parameters t(r).
    """
    intra=np.abs(hops[:,4]-hops[:,7])==0.0
    inter=np.invert(intra)
    l1=np.array([1 if i<N else -1 for i in hops[inter,0]])
    l2=np.array([1 if 2*N<=j<3*N else -1 for j in hops[inter,1]])
    t=np.zeros(hops.shape[0])
    t[intra]=t_intra(hops[intra,2],hops[intra,3],hops[intra,5],hops[intra,6])
    t[inter]=t_inter(hops[inter,2],hops[inter,3],hops[inter,5],hops[inter,6],l1,l2,theta)
    t_onsite=t_intra(0,0,hops_onsite[:,0],hops_onsite[:,1])
    return t,t_onsite



def hamiltonian(hops,hops_onsite,t,t_onsite,N,interlayer=0.0,V=0.0):
    """
    Hamiltonian matrix H(k) as a function of a point k=(kx,ky,kz) in reciprocal space.
    """
    #off-diag hoppings
    i=hops[:,0].astype('int')
    j=hops[:,1].astype('int')
    r1=hops[:,2:5]
    r2=hops[:,5:8]
    tt=np.copy(t)
    inter=np.abs(hops[:,4]-hops[:,7])!=0.0
    tt[inter]=tt[inter]*interlayer
    #onsite hoppings
    I4N=sparse.eye(4*N)    
    #electric-field
    Vdiag=V*np.concatenate((np.ones(2*N),-np.ones(2*N)))
    Vmat=sparse.diags(Vdiag)
    #full matrix
    def H(k):
        data=tt*np.exp(1j*(r2-r1)@k)
        off_diag=sparse.coo_matrix((data, (i, j)),shape=(4*N,4*N))
        off_diag.eliminate_zeros()
        onsite=I4N * np.sum(t_onsite * np.exp(1j*hops_onsite@k))
        return onsite + off_diag + off_diag.getH() + Vmat/2
    return H



def eigenenergies(Hk):
    """
    Only eigenenergies, for small matrices.
    """
    return linalg.eigh(Hk,eigvals_only=True)



def eigenenergies_sparse(Hk,nbands,Ef):
    """
    Only eigenenergies, for big sparse matrices (scipy.sparse).
    """
    e=sparse_linalg.eigsh(Hk,k=nbands,sigma=Ef,return_eigenvectors=False)
    e.sort()
    return e



def eigenstates(Hk):
    """
    Calculates eigenstates, for small matrices.
    """
    return linalg.eigh(Hk,eigvals_only=False)



def eigenstates_sparse(Hk,nbands,Ef):
    """
    Calculates eigenstates, for big sparse matrices (scipy.sparse).
    """
    e=sparse_linalg.eigsh(Hk,k=nbands,sigma=Ef,return_eigenvectors=True)
    e.sort()
    return e



def kticks(pts_per_line_segment):
    """
    K-point path index ticks for plots (might be better understood by seeing the code for band structure plots in other scripts).
    """
    ticks=np.array([sum(pts_per_line_segment[0:i])%sum(pts_per_line_segment) for i in range(len(pts_per_line_segment)+1)])
    ticks[-1]+=sum(pts_per_line_segment)-1
    return ticks



def kpath(kpts,pts_per_line_segment,endpoint=False):
    """
    Parameterized path (MAT-36, remember?) of k-points in reciprocal space.
    
    gamma,ell,ticks=kpath(kpts,pts_per_line_segment)
    
    plt.xticks(ell[ticks],['G','K','M' ...])
    """
    gamma=[] #path of kpoints in R² space (remember MAT-36)
    ell=[] #lengths of the length parameterized path (again, remember MAT-36)
    for n in range(len(kpts)-1):
        if n==len(kpts)-2:
            t=np.linspace(0,1,pts_per_line_segment[n],endpoint=endpoint).reshape(-1,1)
        else:
            t=np.linspace(0,1,pts_per_line_segment[n],endpoint=False).reshape(-1,1)
        gamma_n=kpts[n]+t*(kpts[n+1]-kpts[n])
        ell_n=np.linalg.norm(gamma_n-kpts[n],axis=-1)
        if n>=1:
            ell_n+=ell[-1][-1]+(ell[-1][-1]-ell[-1][-2])
        gamma.append(gamma_n)
        ell.append(ell_n)
    gamma=np.concatenate(gamma)
    ell=np.concatenate(ell)
    return gamma,ell,kticks(pts_per_line_segment) #gamma,ell,index ticks



def bands(H,gamma):
    """
    Electronic bands, for small matrices.
    
    Also works for big matrices, but may be too slow.
    """
    return np.array([eigenenergies(H(k).toarray()) for k in gamma])



def bands_sparse(H,gamma,nbands,Ef):
    """
    Electronic bands, for big matrices.
    """
    return np.array([eigenenergies_sparse(H(k),nbands,Ef) for k in gamma])



def bands_with_layer_character(H,gamma):
    """
    Electronic bands, for small matrices. Calculates the layer character for colormap plots.
    
    Also works for big matrices, but may be too slow.
    """
    e=[]
    c1=[]
    c2=[]
    N=H(np.zeros(3)).shape[0]//4
    for k in gamma:
        ek,psik=eigenstates(H(k).toarray())
        psik1=psik[0:2*N,:]
        psik2=psik[2*N::,:]
        c1k=(np.abs(psik1)**2).sum(axis=0)
        c2k=(np.abs(psik2)**2).sum(axis=0)
        e.append(ek)
        c1.append(c1k)
        c2.append(c2k)
    e=np.array(e)
    c1=np.array(c1)
    c2=np.array(c2)
    return e,c1,c2



def bands_with_layer_character_sparse(H,gamma,nbands,Ef):
    """
    Electronic bands, for big matrices. Calculates the layer character for colormap plots.
    """
    e=[]
    c1=[]
    c2=[]
    N=H(np.zeros(3)).shape[0]//4
    for k in gamma:
        ek,psik=eigenstates_sparse(H(k),nbands,Ef)
        psik1=psik[0:2*N,:]
        psik2=psik[2*N::,:]
        c1k=(np.abs(psik1)**2).sum(axis=1)
        c2k=(np.abs(psik2)**2).sum(axis=1)
        e.append(ek)
        c1.append(c1k)
        c2.append(c2k)
    e=np.array(e)
    c1=np.array(c1)
    c2=np.array(c2)
    return e,c1,c2
    


class TwistedBilayerGraphene:
    def __init__(self,p,q,sparse=None):
        """
        Initializes all relevant constant parameters.
        """
        
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
        
        if N<=100 and not sparse: #rough estimate
            self._bands=lambda H,gamma,nbands,Ef: bands(H,gamma)
            self._bands_with_layer_character=lambda H,gamma,nbands,Ef: bands_with_layer_character(H,gamma)
        else:
            self._bands=bands_sparse
            self._bands_with_layer_character=bands_with_layer_character_sparse
        
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
        
        
    def generate_magnetic_unit_cell(self,q): #this is "irreversible"
        """
        (Experimental) Extended unit cell for inclusion of magnetic field. This is "irreversible".
        """
        self.qmag=q
        x,y,_=self.L2
        theta=np.arctan2(y,x)
        self.r=np.concatenate([self.r+n*self.L1 for n in range(q)])@rot(np.pi/2-theta).T
        self.L2=rot(np.pi/2-theta)@self.L2
        self.L1=rot(np.pi/2-theta)@(q*self.L1)
        self.N=q*self.N
        #K[0],M[0],Kp[0],Mp[0],K[1],...
        L1,L2=self.L1,self.L2
        L3=np.array([0,0,1])
        G1=2*np.pi*np.cross(L2,L3)/np.dot(L1,np.cross(L2,L3))
        G2=2*np.pi*np.cross(L3,L1)/np.dot(L1,np.cross(L2,L3))
        kpt=lambda m,n: m*G1+n*G2
        Gamma=kpt(0,0)
        K=kpt(1/3,-1/3),kpt(1/3,2/3),kpt(-2/3,-1/3)
        Kp=kpt(2/3,1/3),kpt(-1/3,1/3),kpt(-1/3,-2/3)
        M=kpt(1/2,0),kpt(0,1/2),kpt(-1/2,-1/2)
        Mp=kpt(1/2,1/2),kpt(-1/2,0),kpt(0,-1/2)
        self.path_GMKG=[Gamma,M[0],K[0],Gamma]
        self.path_KGMKp=[K[0],Gamma,M[1],Kp[1]]
        self.G1=G1
        self.G2=G2
        self.kpt=kpt
        self.Gamma=Gamma
        self.K=K
        self.Kp=Kp
        self.M=M
        self.Mp=Mp
        #bands calc
        if self.N==1:
            self._bands=lambda H,gamma,nbands,Ef: bands(H,gamma)
        else:
            self._bands=bands_sparse
        
    
    def calc_hops(self,max_distance,t_intra=t_intra,t_inter=t_inter):
        """
        Calculate the hoppings.
        """
        self.hops=hoppings(self.r,self.L1,self.L2,400*self.N,max_distance)
        self.hops_onsite=hoppings_onsite(self.L1,self.L2,max_distance)
        self.t,self.t_onsite=hopping_parameters(self.hops,self.hops_onsite,self.N,self.theta,t_intra,t_inter)


    def include_peierls_substitution(self): #this is "irreversible"
        """
        (Experimental) Change hopping parameters for inclusion of magnetic field. This is irreversible.
        """
        S=np.linalg.norm(np.cross(self.L1,self.L2))
        B=phi0/S
        peierls=lambda hops: (2*np.pi/phi0)*(B/2)*(hops[:,5]+hops[:,2])*(hops[:,6]-hops[:,3])
        self.t=self.t*np.exp(1j*(2*np.pi/phi0)*(B/2)*(self.hops[:,5]+self.hops[:,2])*(self.hops[:,6]-self.hops[:,3]))
        #for now, we ignore changes at periodic repetitions ("on-site")
        self.B=B
                
        
    def set_hamiltonian(self,interlayer=0.0,V=0.0):
        """
        Define the hamiltonian matrix H(k) as a function of a point k=(kx,ky,kz) in reciprocal space.
        """
        self.H=hamiltonian(self.hops,self.hops_onsite,self.t,self.t_onsite,self.N,interlayer,V)
        

    def set_kpath(self,kpts,pts_per_line_segment,endpoint=False):
        """
        Define the parametrized path (MAT-36, remember?) of k-points in reciprocal space.
        
        Some convenient constants can be used (kpts=self.path_GMKG, self.path_KGMKp).
        """
        self.gamma,self.ell,self.kticks=kpath(kpts,pts_per_line_segment,endpoint)

        
    def calc_bands(self,nbands,Ef):
        """
        Calcule the electronic bands (eigenenergies). Must set hamiltonian (self.set_hamiltonian) and kpath (self.set_kpath) before.
        """
        self.bands=self._bands(self.H,self.gamma,nbands,Ef)


    def calc_bands_and_layer_characters(self,nbands,Ef):
        """
        Calcule the electronic bands (eigenenergies) *and* eigenfunctions. Must set hamiltonian (self.set_hamiltonian) and kpath (self.set_kpath) before.
        """
        self.bands,self.layer1_character,self.layer2_character=self._bands_with_layer_character(self.H,self.gamma,nbands,Ef)
        