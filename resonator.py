from select import select
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares,minimize, LinearConstraint, NonlinearConstraint
import os
import time
#%%
fontName = 'Times New Roman'
fontSize = 12
plt.rc('font', **{'family': 'serif', 'serif': [fontName], 'size': fontSize})
plt.rc('mathtext', **{'default': 'regular'})
plt.rc('text', **{'usetex': False})
plt.rc('lines', **{'linewidth': 2})

#%%

class resonator():

    def __init__(self,a_n,L_n,a_c,L_c):
        '''
        This function initializes an instance of a "helmholtz" object with the following parameters. 

        Parameter:
        a_n: Neck radius [m]
        L_n: Neck length [m]
        a_c: Cavity radius [m]
        L_c: Cavity length [m]
        '''
        self.a_n = a_n
        self.L_n = L_n
        self.a_c = a_c
        self.L_c = L_c

        self.A_n = np.pi*a_n**2
        self.A_c = np.pi*a_c**2


    def set_Z(self,f,WG = True,rad = True,loss = False):
        '''
        This function computes the complex normal impedance of the resonator. This can be accomplished using two different techniques. In the basic acoustic element (BAE) approach
        the resonator is modeled as a compination of acoustic masses and compliance elements. This technique is not valid at very high frequencies, therefore, only the first several
        resonance peaks would be resolved. The second option would be to use the waveguide (WG) technique, in which the values of the acoustic mass and compliance elements are derived 
        from the governing equations of fluid dynamics. Therefore, this is considered the exact solution and will resolve all resonance peaks. 
        
        Parameters: 
        f: single or an array of frequency values [Hz]
        WG: set equal to True to compute the impedance using the waveguide solution or False to use the basic acoustic element approach.
        rad: set equal to True in order to include the spherical radiation impedance.
        '''

        self.w = 2*np.pi*f
        k = self.w/c
        Z0_n = rho*c/self.A_n
        Z0_c = rho*c/self.A_c

        if WG: 
            self.Za_n =  1j*Z0_n*np.tan(k*self.L_n/2)
            self.Zb_n = Z0_n/(1j*np.sin(k*self.L_n))

            self.Za_c  = 1j*Z0_c*np.tan(k*self.L_c/2)
            self.Zb_c  = Z0_c/(1j*np.sin(k*self.L_c))

        else:

            self.Za_n =  1j*k*rho*c*self.L_n/(2*self.A_n)
            self.Zb_n = rho*c/(1j*k*self.A_n*self.L_n)

            self.Za_c =  1j*k*rho*c*self.L_c/(2*self.A_c)
            self.Zb_c  = rho*c/(1j*k*self.A_c*self.L_c)

            if loss:
                
                # viscous boundary layer thickness for air @ 20C [m]
                del_mu = np.sqrt(2*18e-6/(rho*self.w))
                # thermal boundary layer thickness for air @ 20C [m]
                del_k = np.sqrt(2*0.026/(rho*self.w*1006))
                # ratio of specific heats for air @ 20C
                gamma = 1.4

                # wetted area
                P_n = 2*np.pi**a_n*L_n
                P_c = 2*np.pi**a_c*L_c

                R_mu_n = rho*self.w*del_mu*P_n/(2*self.A_n**2)
                R_mu_c = rho*self.w*del_mu*P_c/(2*self.A_c**2)
                
                R_k_n = ((gamma-1)*self.w*del_k*P_n/(2*rho*c**2))**-1
                R_k_c =  ((gamma-1)*self.w*del_k*P_c/(2*rho*c**2))**-1

                # Z_c_2 = (np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[self.Zb_c**-1+R_k_c**-1,np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))
                # Z_c_2 = np.array([[np.ones(len(self.w)),R_mu_c/2],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1)@np.array([[np.zeros(len(self.w)),self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1)@ \
                # np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[1/R_k_c,np.ones(len(self.w))]]).transpose(-1,0,1)@np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[1/self.Zb_c,np.ones(len(self.w))]]).transpose(-1,0,1)

        if isinstance(self.w,np.ndarray):

            if loss:
                self.T_n = (np.array([[np.ones(len(self.w)),R_mu_n/2+self.Za_n],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[self.Zb_n**-1+R_k_n**-1,np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),R_mu_n/2+self.Za_n],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))
                self.T_c = (np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[self.Zb_c**-1+R_k_c**-1,np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))

            else:
                self.T_n = np.array([[1+self.Za_n/self.Zb_n,2*self.Za_n+self.Za_n**2/self.Zb_n],[1/self.Zb_n,1+self.Za_n/self.Zb_n]]).transpose(-1,0,1)
                # self.T_c = np.array([[1+self.Za_c/self.Zb_c,self.Za_c],[1/self.Zb_c,np.ones(len(self.w))]]).transpose(-1,0,1)
                self.T_c = np.array([[1+self.Za_c/self.Zb_c,2*self.Za_c+self.Za_c**2/self.Zb_c],[1/self.Zb_c,1+self.Za_c/self.Zb_c]]).transpose(-1,0,1)

        else:
            if loss:
                self.T_n = (np.array([[1,R_mu_n/2+self.Za_n],[0,1]]).transpose(-1,0,1))@(np.array([[1,0],[self.Zb_n**-1+R_k_n**-1,1]]).transpose(-1,0,1))@(np.array([[1,R_mu_n/2+self.Za_n],[0,1]]).transpose(-1,0,1))
                self.T_c = (np.array([[1,R_mu_c/2+self.Za_c],[0,1]]).transpose(-1,0,1))@(np.array([[1,0],[self.Zb_c**-1+R_k_c**-1,1]]).transpose(-1,0,1))@(np.array([[1,R_mu_c/2+self.Za_c],[0,1]]).transpose(-1,0,1))

            else:
                self.T_n = np.array([[1+self.Za_n/self.Zb_n,2*self.Za_n+self.Za_n**2/self.Zb_n],[1/self.Zb_n,1+self.Za_n/self.Zb_n]])
                self.T_c = np.array([[1+self.Za_c/self.Zb_c,self.Za_c],[1/self.Zb_c,1]])

        if rad:
            A_rad = 4*self.A_n
            a_rad = 2*self.a_n
            Z_rad = rho*c/A_rad*(1j*k*a_rad/(1+1j*k*a_rad))

            if isinstance(self.w,np.ndarray):
                self.T_rad = np.array([[np.ones(len(self.w)),Z_rad],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1)
            else:
                self.T_rad = np.array([[1,Z_rad],[0,1]])

            self.T = self.T_rad@self.T_n@self.T_c
        else:
            self.T = self.T_n@self.T_c

        if isinstance(self.w,np.ndarray):
            self.Z = self.T[:,0,0]/ self.T[:,1,0]
        else:
            self.Z = self.T[0,0]/ self.T[1,0]
        self.set_alpha()

    
    def get_Z(self):
        assert hasattr(self,'Z'), 'Impedance has not been computed, run the set_Z(f) function first.'
        return self.Z

    def get_alpha(self):
        assert hasattr(self,'Z'), 'The absorption coefficient has not been computed, run the set_Z(f) function first.'
        return self.alpha


    def set_alpha(self, f=[]):
        '''
        This function returns the normal impedance absorption coefficient for the helmholtz resonator. The impedance must be computed prior to the absorption coefficient. 
        '''
        # assert hasattr(self,'Z'), 'Impedance has not been computed, run the get_Z(w) before computing the absorption coefficient.'
        
        self.alpha = 1 - abs((self.Z/(c*rho)-1)/(self.Z/(c*rho)+1))**2

    # def get_p(self,p_in):
    #     U = p_in/self.Z
    #     np.linalg.inv(self.Z)@np.array([[p_in],[U]])


    def minimize_Z(self,f,lb=[0,0,0,0,0,0],ub=[np.inf,np.inf,np.inf,np.inf,np.inf,1]):
        '''
        This wrapper function performs a constrained optimization of the resonator which determines the dimensions of the resonator that minimizes the reflection coefficient, R
        for a particular resonance frequency. There are three primary constraint conditions: a_n & a_c > 0, L_n & L_c > 0, L_n/L_c <1. The lower and upper bounds for each condition
        can be specified as lists. Where the first four elements correspond to the min and max values of a_n, a_c, L_n, and L_c. The fourth element of the upper bound list 
        sets the maximum length of the resonator (L_n+L_c). The final element simply states that the neck radius must be less than that of the cavity, a_n < a_c.

        The optimizer changes the geometrical attributes (a_c,a_n,L_c,L_n) of the resonator instance during each itteration. 
        
        parameters:
        lb: list of six lower bounds that correspond to each of the constraints. 
        ub: list of six upper bounds that correspond to each of the constraints. 

        '''

        def get_constraints(lb,ub):
            constraints = NonlinearConstraint(fun = lambda x: [x[0],x[2],x[1],x[3],x[1]+x[3],x[0]/x[2]],lb =lb,ub = ub)
            return constraints

        def opt_wrap(dims,f):
            self.a_n, self.L_n, self.a_c,self.L_c = dims[0],dims[1], dims[2],dims[3]
            self.set_Z(f,WG=False)
            self.R = abs((self.Z/(c*rho)-1)/(self.Z/(c*rho)+1))**2
            # print(f'Searching for optimal geometry: R = {round(R,2)}, a_n = {round(self.a_n,2)}, L_n = {round(self.a_n,2)}, a_c = {round(self.a_n,2)}, L_c = {round(self.a_n,2)}')
            print(f'Searching for optimal geometry: R = {self.R}, a_n = {a_n}, L_n = {L_n}, a_c = {a_c}, L_c = {L_c}')

            return self.R

        minimize(opt_wrap,x0 = [self.a_n,self.L_n,self.a_c,self.L_c],constraints = get_constraints(lb,ub),args = f,method = 'trust-constr')

    def plot(self,xlim = [1e2,10e3]):
        
        fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15)
        ax[0].tick_params(axis = 'x', labelsize=0)
        ax[0].loglog(self.w/(2*np.pi),np.real(self.Z**-1))
        ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
        ax[0].set_xlim(xlim)
        ax[0].grid()

        ax[1].tick_params(axis = 'x', labelsize=0)
        ax[1].loglog(self.w/(2*np.pi),np.imag(self.Z**-1))
        ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
        ax[1].set_xlim(xlim)
        ax[1].grid()

        ax[-1].plot(self.w/(2*np.pi),self.alpha)
        ax[-1].set_xscale('log')
        ax[-1].set_xlim(xlim)
        ax[-1].set_ylabel(r'$\alpha$')
        ax[-1].set_xlabel('Frequency [Hz]')
        ax[-1].grid()
        plt.show()


#%%
# speed of sound [m/s]
c = 343
df = 1
f = np.arange(1,10e3/df+1)*df
#   density [kg/m3]
rho = 1.25

#%%

a_n ,a_c,L_n , L_c = 0.01,0.05,0.05,0.05
helm1 = resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)

f0 = c/(4*(L_c+L_c))
helm1.set_Z(f,WG = False,rad = True,loss = True)
Z_WG = helm1.Z
alpha_WG = helm1.alpha


helm1.set_Z(f,WG = False,rad = True,loss=False)
Z_BAE = helm1.Z
alpha_BAE = helm1.alpha



f0  = 250
helm1.minimize_Z(f0)
helm1.set_Z(f,WG = True)
Z_WG = helm1.Z
alpha_WG = helm1.alpha

helm1.set_Z(f,WG = False)
Z_BAE = helm1.Z
alpha_BAE = helm1.alpha


#%%

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].loglog(f,np.real(Z_WG**-1))
# ax[0].loglog(f,np.real(Z_BAE**-1))
ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
ax[0].set_xlim([100,10e3])
ax[0].grid()


ax[1].set_xlim([100,10e3])
ax[1].tick_params(axis = 'x', labelsize=0)
ax[1].loglog(f,np.imag(Z_WG**-1))
# ax[1].loglog(f,np.imag(Z_BAE**-1))
ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
ax[1].grid()

ax[-1].plot(f,abs((helm1.T_c[:,0,0]/helm1.T_c[:,1,0])/helm1.Z))
# ax[-1].plot(f,(helm1.T[:,0,0]**-1))

# ax[-1].plot(f,alpha_BAE)
ax[-1].set_xscale('log')
ax[-1].set_xlim([100,10e3])
ax[-1].set_ylabel(r'$\alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].legend(['Waveguide','Basic Acoustic Element'])

# ax[-1].set_xlim([100,10e3])
# ax[-1].tick_params(axis = 'x', labelsize=0)
# ax[-1].plot(w/(2*np.pi),np.abs(Z_c/Z))
# ax[-1].set_yscale('log')
# ax[-1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# ax[-1].grid()

# ax[-1].loglog(w/(2*np.pi),np.angle(Z**-1)*180/np.pi)
# ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae**-1)*180/np.pi)
# ax[-1].plot(w/(2*np.pi),alpha)
# ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae**-1)*180/np.pi)

ax[-1].plot(f,alpha_WG)
ax[-1].plot(f,alpha_BAE)

ax[-1].set_xscale('log')
ax[-1].set_xlim([100,10e3])
ax[-1].set_ylabel(r'$\alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].legend(['Waveguide','Basic Acoustic Element'])



# #%%
# # natural frequency [Hz]
# f0 = 750
# w0 = 2*np.pi*f0
# k0 = w0/c

# #Airfoil cross-sectional area [m^2] (upper bound of constraint)
# XsectA = 0.00064214
# # Blade radius - root cutout [m^2]
# R = 0.78364

# A_n,L_n,A_c,L_c = XsectA/2,R/2,XsectA/2,R/2

# # add constraint for length of neck and cavity = cannot be negative
# # con = LinearConstraint([[1,0,0,0],[0,0,1,0],[0,1,0,1],[0,1,0,0],[0,0,0,1]],lb = [1e-3*XsectA,1e-3*XsectA,0,1e-7,1e-7],ub = [XsectA,XsectA, R,np.inf ,np.inf  ])
# con = NonlinearConstraint(fun = con_fun,lb = [0,0,0,0,0,0],ub = [XsectA,XsectA,np.inf,np.inf,R,1])
# # res = least_squares(opt_wrap,x0 = [A_n,L_n,A_c,L_c],bounds = [0,1],args = [w0])
# res = minimize(opt_wrap,x0 = [A_n,L_n,A_c,L_c],constraints = con,args = w0,method = 'trust-constr')

# print(res.x)
# A_n,L_n,A_c,L_c = res.x
# R_n = np.sqrt(A_n/np.pi)
# R_c = np.sqrt(A_c/np.pi)

# Z = Z_helmholtz(A_n,L_n,A_c,L_c,w,loss=False)
# Z_bae = Z_helmholtz(A_n,L_n,A_c,L_c,w,wg = False,loss = False)

# R = (Z-1)/(Z+1)
# alpha = 1-np.abs(R)**2

# #%%

# fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15)
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].loglog(w/(2*np.pi),np.real(Z[0]**-1))
# ax[0].loglog(w/(2*np.pi),np.real(Z_bae[0]**-1))
# ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
# ax[0].set_xlim([100,10e3])
# ax[0].grid()

# ax[1].set_xlim([100,10e3])
# ax[1].tick_params(axis = 'x', labelsize=0)
# ax[1].loglog(w/(2*np.pi),np.imag(Z[0]**-1))
# ax[1].loglog(w/(2*np.pi),np.imag(Z_bae[0]**-1))
# ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# ax[1].grid()

# ax[-1].set_xlim([100,10e3])
# ax[-1].loglog(w/(2*np.pi),np.angle(Z[0]**-1)*180/np.pi)
# ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae[0]**-1)*180/np.pi)
# ax[-1].set_ylabel('$Phase \ [\circ]$')
# ax[-1].set_xlabel('Frequency [Hz]')
# ax[-1].grid()
# ax[-1].set_xlim([100,10e3])
# ax[-1].set_ylim([-180, 180])
# ax[-1].legend(['Waveguide','Basic Acoustic Element'])

# ax[-1].set_xlim([100,10e3])
# ax[-1].loglog(w/(2*np.pi),np.angle(Z[0]**-1)*180/np.pi)
# ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae[0]**-1)*180/np.pi)
# ax[-1].set_ylabel('$Phase \ [\circ]$')
# ax[-1].set_xlabel('Frequency [Hz]')
# ax[-1].grid()
# ax[-1].set_xlim([100,10e3])
# ax[-1].set_ylim([-180, 180])
# ax[-1].legend(['Waveguide','Basic Acoustic Element'])


# plt.show()
