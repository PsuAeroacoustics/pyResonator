#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.interpolate import RectBivariateSpline,CubicSpline
from scipy.special import jv
from scipy.optimize import least_squares

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 12

#%%

def newton(fun,fun_prime,x0,toll = 5e-10):
    err = 1
    while np.any(err > toll):
        x1 = x0-fun(x0)/fun_prime(x0)
        err = abs((x1-x0)/x1)
        # print(np.max(err))
        x0 = x1
    return x1


class fs():
    def __init__(self,t,r,phi,c = 340,rho = 1.125,nu = 14.88e-6):
        '''
        This function initializes an instance of a "fs" or facesheet object with the following parameters. 
        Parameter:
        t: facesheet plate thickness [m]
        r: radius of opening [m]
        phi: perforation rate (porosity: total area of openings/area of sample) 

        '''
        # facesheet thickness [m]
        self.t = t
        # radius of holes in the facesheet [m]
        self.r = r
        # perforation rate (porosity) - cavity open area ratio
        self.phi = phi

        # speed of sound [m/s]
        self.c = c
        # density [kg/m^3]
        self.rho = rho
        # kinematic viscosity [m^2/s]
        self.nu = nu
        

    def set_Z(self,f,model = '2P',SPL = 0,M = 0, Z_cav = 0):
        '''
        This function computes the complex normal acoustic impedance and absorption of the facesheet. The user has the choice to use the 2-parameter or the Atalla and Sgard semiempirical model. The default is the 2-parameter model. The Atalla and Sgard model simplifies a generalized model (JCAPL model) by limiting it to the specific case of a perforated facesheet.
        The model essentially treats the facesheet as an equivalent fluid subject to visco-inertial losses associated with the dissipative effects of the viscous boundary layer and flow distortions (resistive component) within the openings of the facesheet as well as the mass of the 
        air moving through that opening (reactive component). 
        [Jones, Michael, Tony Parrott, and Willie Watson. "Uncertainty and sensitivity analyses of a two-parameter impedance prediction model." 14th AIAA/CEAS Aeroacoustics Conference (29th AIAA Aeroacoustics Conference). 2008.]
        [Atalla, Noureddine, and Franck Sgard. "Modeling of perforated plates and screens using rigid frame porous models." Journal of sound and vibration 303.1-2 (2007): 195-208.]
        '''

        self.w = 2*np.pi*f
        self.k = self.w/self.c
        if model == 'AS':
            # Geometric torosity with end correction
            alpha_inf = 1+2/self.t*(0.48*np.sqrt(np.pi*self.r**2)*(1-1.14*np.sqrt(self.phi)))
            # Flow resistivity
            sigma = 8*self.nu*self.rho/(self.phi*self.r**2)
            G = np.sqrt(1+1j*4*self.w*self.rho*alpha_inf**2*self.nu*self.rho/(sigma**2*self.phi**2*self.r**2))
            # Effective density
            rho_e = self.rho*alpha_inf*(1+sigma*self.phi*G/(1j*self.w*self.rho*alpha_inf))
            self.Z = 1j*self.w*rho_e*self.t/(self.c*self.rho*self.phi)
            self.alpha = 1 - abs((self.Z-1)/(self.Z+1))**2

        else:
            
            Cd = 0.76
            delta_BL = 1.75e-3
            Ki = 0.5 
            Ke = 0.5
            # A = 64*self.nu*self.t/(2*self.c*self.phi*Cd*(2*self.r)**2)
            # B = (Ki+Ke)/(2*self.c*(self.phi*Cd)**2)

            A = 64*self.rho*self.nu*self.t/(2*self.phi*Cd*(2*self.r)**2)
            B = self.rho*(Ki+Ke)/(2*(self.phi*Cd)**2)

            # A = 1.4
            # B = 0.2336
            p = 20e-6*10**(SPL/20)

            epsilon = 0.85*(1-0.7*np.sqrt(self.phi))/(1+305*M**3)

            th_gf = M/(self.phi*(2+1.256*delta_BL/(2*self.r)))
            Xm = (self.k*(self.t+2*epsilon*self.r))/(self.phi*Cd)

            # fun  = lambda x: x - A-B*p/(self.rho*self.c*((x+np.real(Z_cav))**2+(Xm+np.imag(Z_cav))**2)**(1/2))-th_gf
            # fun_prime = lambda x: 1 +B*p*(x+np.real(Z_cav))/(self.rho*self.c*((x+np.real(Z_cav))**2+(Xm+np.imag(Z_cav))**2)**(3/2))

            fun  = lambda x: self.rho*self.c*x - A-B*p/(self.rho*self.c*((x+np.real(Z_cav))**2+(Xm+np.imag(Z_cav))**2)**(1/2))-th_gf*self.rho*self.c
            fun_prime = lambda x: self.rho*self.c +B*p*(x+np.real(Z_cav))/(self.rho*self.c*((x+np.real(Z_cav))**2+(Xm+np.imag(Z_cav))**2)**(3/2))


            R = newton(fun = fun,fun_prime =fun_prime,x0 = 1,toll = 5e-5 )
            # print(f'v_rms = {np.mean(p/(self.rho*self.c*((R+np.real(Z_cav))**2+(Xm+np.imag(Z_cav))**2)**(1/2)))}')
            self.Z = R+1j*Xm


    def get_Z(self):
        assert hasattr(self,'Z'), 'Impedance has not been computed, run the set_Z(f) function first.'
        return self.Z

    def get_alpha(self):
        assert hasattr(self,'Z'), 'The absorption coefficient has not been computed, run the set_Z(f) function first.'
        return self.alpha




class resonator():

    def __init__(self,c = 340,P = 101325,gamma =1.4,rho = 1.125,nu = 14.88e-6,Pr = 0.71,q=1.30384,s_b = 1.37,t = 0.01,sigma = None,phi = None,a_n = None,L_n = None,a_c = None,L_c= None):
        '''
        This function initializes an instance of a "resonator" object with the following parameters. The required parameters vary depending on w 

    General fluid properties:
        c: speed of sound [m/s]
        P: atmospheric pressure [Pa]
        rho: density [kg/m^3]
        nu: kinematic viscosity [m^2/s]
        Pr: Prantl number 
        gamma: ratio of specific heats

    Parameters of helmholtz resonators:
        a_n: Neck radius [m]
        L_n: Neck length [m]
        a_c: Cavity radius [m]
        L_c: Cavity length [m]

    Parameters of bulk porous materials:
        t: thickness [m]
        phi: porosity - ratio of the volume of air in the pores to the total volume
        q: tortuosity - describes the shape of pores - (1/cos(theta) where theta is the pore-slant angle)
        sigma: flow resistivity [MKS rayl/m = Pa*s*m^-2 = N*s*m^-4]
        s_b: shape factor - adjustable parameter describing how much the pore geometry deviates from being cylinderical with a constant cross sectional area
        '''
        if a_n is not None:
            # self.L_n = L_n+(a_n*8/(3*np.pi)*(1-1.25*np.sqrt(0.0225)))
            self.L_n = L_n
            self.L_c = L_c
            self.a_n = a_n
            self.a_c = a_c

            self.A_n = np.pi*a_n**2
            self.A_c = np.pi*a_c**2
        
        # ambient pressure [Pa]
        self.P = P
        # speed of sound [m/s]
        self.c = c
        # density [kg/m^3]
        self.rho = rho
        # kinematic viscosity [m^2/s]
        self.nu = nu
        # Prantl number of air at 20C
        self.Pr = Pr
        # ratio of specific heats 
        self.gamma = gamma

        # thickness [m]
        self.t = t
        # porosity - ratio of the volume of air in the pores to the total volume
        self.phi = phi
        # tortuosity - describes the shape of pores - (1/cos(theta) where theta is the pore-slant angle)
        self.q = q
        # flow resistivity [MKS rayl/m = Pa*s*m^-2 = N*s*m^-4]
        self.sigma = sigma
        # shape factor - adjustable parameter describing how different the pore geometry is from cylinderical with a constant cross sectional area
        self.s_b = s_b

    def set_Z(self,f,model = 'Kirchoff',rad = False,loss = False,interior = False, table = False):
        self.f = f
        self.w = 2*np.pi*f
        if self.sigma is None:
            self.ZKTL(model = model,rad = rad,loss = loss,interior = interior, table = table)
        else:
            self.porous_Z()
        self.set_alpha()
        

    def porous_Z(self):
        if self.phi is None:
            self.delany_bazley()
        else:
            self.biot_allard()
        self.Z = self.Z_c*(self.rho*self.c)**-1*np.tanh(self.Gamma*self.t)**-1

    def delany_bazley(self):
        c = self.f*self.rho/self.sigma
        self.Z_c = self.rho*self.c*(1+0.0571*c**-0.754-1j*0.087*c**-0.732)
        self.Gamma =1j*self.w/self.c*(1+0.0978*c**-0.7-1j*0.189*c**-0.595)

    def biot_allard(self):

        lam = self.s_b*(8*self.q**2*self.rho*self.w/(self.sigma*self.phi))**(1/2)
        T_func = lambda x: jv(1,x)/jv(0,x)
        T = T_func(lam*np.sqrt(-1j))
        F = -1/4*lam*np.sqrt(-1j)*T/(1-2*T/(lam*np.sqrt(-1j)))
        rho_d = self.rho*self.q**2-1j*self.phi*self.sigma/self.w*F
        k_d = self.gamma*self.P*(1+2*(self.gamma-1)/(np.sqrt(self.Pr)*lam*np.sqrt(-1j))*T_func(np.sqrt(self.Pr)*lam*np.sqrt(-1j)))**-1
        self.Gamma =1j*self.w*(rho_d/k_d)**(1/2)
        self.Z_c = 1/self.phi*(rho_d*k_d)**(1/2)        

    def tune_params(self,f,val_data_Re,val_data_Im,bnds = ([1e3,0.5,0.5,.2],[1e5,3,3,1])):
        '''
        This function adjusts the parameters of the Biot-Allard 4-parameter model to achieve a close fit with experimental data. 

        Parameters:
        f: Frequency array over which to evaluate the response
        val_data_Re: Validation data of specific resistance (normalized by rho*c) formated as an array with two columns the first is the frequency and the second is the corresponding value of resistance
        val_data_Im: Validation data of specific reactance (normalized by rho*c) formated as an array with two columns the first is the frequency and the second is the corresponding value of reactance
        bnds: Min/Max bounds on the tunable parameters (flow resistivity, tortuosity, shape factor, and porosity)

        '''

        def get_Z(res_params):
            # modifies parameters
            self.sigma,self.q,self.s_b,self.phi = rescale_input(res_params)
            # returns impedance
            self.set_Z(f_interp)
            # computes the error between the measurment and prediction
            residual = np.abs(val_data-self.Z)
            return residual
        
        # 1/3 octave band frequency array
        f_3rd_oct = np.concatenate(((1000/2**(1/3*np.arange(1,19)[::-1])),(1000*2**(1/3*np.arange(0,14)))))
        f_interp = f_3rd_oct[(val_data_Re[0,0]<f_3rd_oct) & (val_data_Re[-1,0]>f_3rd_oct)]

        # interpolates empirical data
        val_data_Re = CubicSpline(val_data_Re[:,0],val_data_Re[:,1])
        val_data_Im = CubicSpline(val_data_Im[:,0],val_data_Im[:,1])
        val_data = val_data_Re(f_interp)+1j*val_data_Im(f_interp)

        # scales input parameters to range from 0-1
        bnds = np.array(bnds)
        scale_input = lambda x: (x-bnds[0])/(bnds[1]-bnds[0])
        rescale_input = lambda x: x*(bnds[1]-bnds[0])+bnds[0]

        # applies least-squares to tune the parameters
        print('Tuning parameters...')
        out = least_squares(get_Z, x0 = scale_input(np.array([self.sigma,self.q,self.s_b,self.phi])))        
        self.sigma,self.q,self.s_b,self.phi = rescale_input(out.x)
        print(f'$Model Tuned! Porosity: {np.round(self.phi,2)},Flow Resistivity: {np.round(self.sigma,2)} MKS Rayls, Shape Factor: {np.round(self.s_b,2)}, Tortuosity: {np.round(self.q**2,2)} $')
        self.set_Z(f)


    def ZKTL(self,model = 'Kirchoff',rad = False,loss = False,interior = False, table = False):
        '''
        This function computes the complex normal impedance of the resonator. This can be accomplished using two different techniques. In the basic acoustic element (BAE) approach
        the resonator is modeled as a compination of acoustic masses and compliance elements. This technique is not valid at very high frequencies, therefore, only the first several
        resonance peaks would be resolved. The second option would be to use the waveguide (WG) technique, in which the values of the acoustic mass and compliance elements are derived 
        from the governing equations of fluid dynamics. Therefore, this is considered the exact solution and will resolve all resonance peaks. 
        
        Parameters: 
        f: single or an array of frequency values [Hz]
        WG: set equal to True to compute the impedance using the waveguide solution or False to use the basic acoustic element approach.
        rad: set equal to True in order to include the spherical radiation impedance.
        loss: only applies for the basic acoustic element (BAE) model and includes the thermoviscous losses when set to True
        interior: applies correction to account for the arangement of the resonators. If a resonator is surrounded by other resonators that have the smae lengths this parameter should be set to True. 
        table: set equal to True in order to use a lookup table to approximate the propagation constants instead of using the low frequency approximation proposed by Zwikker-Kosten. 
        '''

        k = self.w/self.c
        Z0_n = self.rho*self.c/self.A_n
        Z0_c = self.rho*self.c/self.A_c

        if model == 'WG': 
            self.Za_n =  1j*Z0_n*np.tan(k*self.L_n/2)
            self.Zb_n = Z0_n/(1j*np.sin(k*self.L_n))

            self.Za_c  = 1j*Z0_c*np.tan(k*self.L_c/2)
            self.Zb_c  = Z0_c/(1j*np.sin(k*self.L_c))

        elif model == 'BAE':

            self.Za_n =  1j*self.w*self.rho*self.L_n/(2*self.A_n)
            self.Zb_n = self.rho*self.c**2/(1j*self.w*self.A_n*self.L_n)

            self.Za_c =  1j*self.w*self.rho*self.L_c/(2*self.A_c)
            self.Zb_c  = self.rho*self.c**2/(1j*self.w*self.A_c*self.L_c)

            if loss:
                
                # viscous boundary layer thickness for air @ 20C [m]
                del_mu = np.sqrt(2*18e-6/(self.rho*self.w))
                # thermal boundary layer thickness for air @ 20C [m]
                del_k = np.sqrt(2*0.026/(self.rho*self.w*1006))
                # ratio of specific heats for air @ 20C
                self.gamma = 1.4

                # wetted area
                P_n = 2*np.pi*self.a_n*self.L_n
                P_c = 2*np.pi*self.a_c*self.L_c

                R_mu_n = self.rho*self.w*del_mu*P_n/(2*self.A_n**2)
                R_mu_c = self.rho*self.w*del_mu*P_c/(2*self.A_c**2)
                
                R_k_n = ((self.gamma-1)*self.w*del_k*P_n/(2*self.rho*self.c**2))**-1
                R_k_c =  ((self.gamma-1)*self.w*del_k*P_c/(2*self.rho*self.c**2))**-1

                # Z_c_2 = (np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[self.Zb_c**-1+R_k_c**-1,np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))
                # Z_c_2 = np.array([[np.ones(len(self.w)),R_mu_c/2],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1)@np.array([[np.zeros(len(self.w)),self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1)@ \
                # np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[1/R_k_c,np.ones(len(self.w))]]).transpose(-1,0,1)@np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[1/self.Zb_c,np.ones(len(self.w))]]).transpose(-1,0,1)
        else:

            # reduced frequency of neck and cavity 
            ka_n = k*self.a_n
            ka_c = k*self.a_c

            # shear wave number of neck and cavity
            s_n = self.a_n*np.sqrt(self.w/self.nu)
            s_c = self.a_c*np.sqrt(self.w/self.nu)
            
            if table:
                self.s,self.ka,self.gamma_tab = self.get_gamma_tab()
                f_re_gamma = RectBivariateSpline(x = self.s,y = self.ka, z = np.real(self.gamma_tab))
                f_imag_gamma = RectBivariateSpline(x = self.s,y = self.ka, z = np.imag(self.gamma_tab))

                # Use low-frequency approximation within this region [((ka_n<self.ka[1]) & (s_n>self.s[-1]))]

                gamma_n = np.squeeze(np.array(list(map(lambda x,y: f_re_gamma(x = x,y = y)+1j*f_imag_gamma(x = x,y = y) ,s_n,ka_n))))
                gamma_c = np.squeeze(np.array(list(map(lambda x,y: f_re_gamma(x = x,y = y)+1j*f_imag_gamma(x = x,y = y) ,s_c,ka_c))))

            else:
                # propagation constants
                nu_gamma_n = (1+(1.4-1)/1.4*jv(2,1j**(3/2)*self.Pr**(1/2)*s_n)/jv(0,1j**(3/2)*self.Pr**(1/2)*s_n))**-1
                gamma_n = np.sqrt(jv(0,1j**(3/2)*s_n)/jv(2,1j**(3/2)*s_n))*np.sqrt(1.4/nu_gamma_n)
                nu_gamma_c = (1+(1.4-1)/1.4*jv(2,1j**(3/2)*self.Pr**(1/2)*s_c)/jv(0,1j**(3/2)*self.Pr**(1/2)*s_c))**-1
                gamma_c = np.sqrt(jv(0,1j**(3/2)*s_c)/jv(2,1j**(3/2)*s_c))*np.sqrt(1.4/nu_gamma_c)

            # gamma_n = np.squeeze(np.array(list(map(lambda x,y: f_re_gamma(x = x,y = y)+1j*f_imag_gamma(x = x,y = y) ,s_n,ka_n))))
            # gamma_c = np.squeeze(np.array(list(map(lambda x,y: f_re_gamma(x = x,y = y)+1j*f_imag_gamma(x = x,y = y) ,s_c,ka_c))))

            # characteristic impedance
            Zc_n = -1j/gamma_n*jv(0,1j**(3/2)*s_n)/jv(2,1j**(3/2)*s_n)
            Zc_c = -1j/gamma_c*jv(0,1j**(3/2)*s_c)/jv(2,1j**(3/2)*s_c)
            
        if model == 'BAE' or model == 'WG':

            if isinstance(self.w,np.ndarray):
                if loss and model == 'BAE':
                    self.T_n = (np.array([[np.ones(len(self.w)),R_mu_n/2+self.Za_n],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[self.Zb_n**-1+R_k_n**-1,np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),R_mu_n/2+self.Za_n],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))
                    self.T_c = (np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),np.zeros(len(self.w))],[self.Zb_c**-1+R_k_c**-1,np.ones(len(self.w))]]).transpose(-1,0,1))@(np.array([[np.ones(len(self.w)),R_mu_c/2+self.Za_c],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1))

                else:
                    self.T_n = np.array([[1+self.Za_n/self.Zb_n,2*self.Za_n+self.Za_n**2/self.Zb_n],[1/self.Zb_n,1+self.Za_n/self.Zb_n]]).transpose(-1,0,1)
                    # self.T_c = np.array([[1+self.Za_c/self.Zb_c,self.Za_c],[1/self.Zb_c,np.ones(len(self.w))]]).transpose(-1,0,1)
                    self.T_c = np.array([[1+self.Za_c/self.Zb_c,2*self.Za_c+self.Za_c**2/self.Zb_c],[1/self.Zb_c,1+self.Za_c/self.Zb_c]]).transpose(-1,0,1)

            else:
                if loss and model == 'BAE':
                    self.T_n = (np.array([[1,R_mu_n/2+self.Za_n],[0,1]]).transpose(-1,0,1))@(np.array([[1,0],[self.Zb_n**-1+R_k_n**-1,1]]).transpose(-1,0,1))@(np.array([[1,R_mu_n/2+self.Za_n],[0,1]]).transpose(-1,0,1))
                    self.T_c = (np.array([[1,R_mu_c/2+self.Za_c],[0,1]]).transpose(-1,0,1))@(np.array([[1,0],[self.Zb_c**-1+R_k_c**-1,1]]).transpose(-1,0,1))@(np.array([[1,R_mu_c/2+self.Za_c],[0,1]]).transpose(-1,0,1))

                else:
                    self.T_n = np.array([[1+self.Za_n/self.Zb_n,2*self.Za_n+self.Za_n**2/self.Zb_n],[1/self.Zb_n,1+self.Za_n/self.Zb_n]])
                    self.T_c = np.array([[1+self.Za_c/self.Zb_c,self.Za_c],[1/self.Zb_c,1]])
        else:
            
            self.T_n = np.array([[np.cosh(k*gamma_n*self.L_n),Zc_n*np.sinh(k*gamma_n*self.L_n)],[Zc_n**-1*np.sinh(k*gamma_n*self.L_n),np.cosh(k*gamma_n*self.L_n)]]).transpose(-1,0,1)
            self.T_c = np.array([[np.cosh(k*gamma_c*self.L_c),Zc_c*np.sinh(k*gamma_c*self.L_c)],[Zc_c**-1*np.sinh(k*gamma_c*self.L_c),np.cosh(k*gamma_c*self.L_c)]]).transpose(-1,0,1)

        if rad:
            
            if interior:
                A_rad = self.A_n
                a_rad = self.a_n
            else:
                A_rad = 4*self.A_n
                a_rad = 2*self.a_n
            
            Br = 0.5
            Bi = 8/(3*np.pi)
            Z_rad = (1j*k*a_rad/(1+1j*k*a_rad))*A_rad*self.rho*self.c
            # Z_rad = (1j*self.w*self.rho*a_rad/(1+1j*k*a_rad))*(self.c*self.rho)**-1


            # Z_rad = (Br*(k*a_rad)**2+1j*Bi*(k*a_rad))

            if isinstance(self.w,np.ndarray):
                self.T_rad = np.array([[np.ones(len(self.w)),Z_rad],[np.zeros(len(self.w)),np.ones(len(self.w))]]).transpose(-1,0,1)
            else:
                self.T_rad = np.array([[1,Z_rad],[0,1]])

            self.T = self.T_rad@self.T_n@self.T_c
        else:
            self.T = self.T_n@self.T_c

        #imparts impermeable boundary condition on the wall of resonator
        self.P,self.U = (self.T@[1,0]).transpose()

        if model == 'BAE' or model == 'WG':
            # nondimensionalizes impedance of waveguide or basic acoustic element model by (rho*a0)
            self.Z = self.P/(self.U/self.A_n)*(self.rho*self.c)**-1
        else:
            self.Z = self.P/self.U
            # Z2 = -1j*np.tan(-k*gamma_n*1j*self.L_n)**-1+-1j*np.tan(-k*gamma_c*1j*self.L_c)**-1

        
    def get_Z(self):
        assert hasattr(self,'Z'), 'Impedance has not been computed, run the set_Z(f) function first.'
        return self.Z

    def get_alpha(self):
        assert hasattr(self,'Z'), 'The absorption coefficient has not been computed, run the set_Z(f) function first.'
        return self.alpha

    def set_alpha(self):
        '''
        This function returns the normal impedance absorption coefficient for the helmholtz resonator. The impedance must be computed prior to the absorption coefficient. 
        '''
        # assert hasattr(self,'Z'), 'Impedance has not been computed, run the get_Z(w) before computing the absorption coefficient.'
        
        self.alpha = 1 - abs((self.Z-1)/(self.Z+1))**2


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

    def set_vdata(self,vdat_file):
        '''
        This function reads in and formats the validation data for the resonator. The data must be formatted as a text files with the first column being the frequency vector and the second the frequency response. 
        parameters: 
        vdat_file: The absolute path to the .txt file that contains the validation data. 
        '''
        with open(vdat_file) as f:
            vdat = np.array(re.split( ',|\n',f.read()))[:-1].astype(float)
        self.vdat = vdat.reshape(int(len(vdat)/2),2)
            
    @staticmethod
    def get_gamma_tab():
        ''' 
        This function imports and formats the tables of the real and imaginary components of the propagation constant of plane waves in cylinders as reported by Tijdeman. The propagation constants 
        include the effects of thermal and viscous losses. The values in these tables were determined by solving Kirchoff's original governing equation iteratively, therefore, these values are exact. 
        The columns of the table correspond to various values of the reduced frequency (ka), while each rows corresponds to different values of the shear wave number (s). Note that the values in the 
        first column were obtained by Zwikker and Kosten low-frequency approximation of the solution (ka<<1).
        (Tijdeman, H. "On the propagation of sound waves in cylindrical tubes." Journal of Sound and Vibration 39.1 (1975): 1-33.) 
        
        parameters:
        s: shear wave number. 
        ka: reduced frequency. 
        data: table of real and imaginary components of the propagation constants. 

        '''
        f_dir = os.path.join(os.getcwd(),'Tijdeman_gamma')
        f_name = ['re_gamma.txt','imag_gamma.txt']

        data_temp = np.empty((len(f_name),36,11))
        for i,n in enumerate(f_name):
            with open(os.path.join(f_dir,n)) as f:
                data_temp[i] = np.array(re.split("\t|\n",f.read())).reshape((36,11)).astype(float)
        s = data_temp[0,1:,0]
        ka = data_temp[0,0,1:]

        data = data_temp[0,1:,1:]+1j*data_temp[1,1:,1:]
        del data_temp
        
        return s,ka,data


#%%
# vdat_file = '/Users/danielweitsman/Documents/research/BVI_helmholtz/probe_data.csv'
# vdat_file = '/Users/danielweitsman/Downloads/probe_data_rad.csv'
# with open(vdat_file) as f:
#     vdat = np.array(re.split( ",|\n", f.read().replace('i','j')))[:-1]
#     vdat = vdat.reshape(int(len(vdat)/3),3)
#     vdat = vdat.astype(complex)

# c = 340
# f = np.real(vdat[:,0])
# k = 2*np.pi*f/c
# N = len(vdat)
# s = 0.03175
# l = 0.09525
# rho = 1.125
# S_12 = np.real(np.conj(vdat[:,1])*vdat[:,-1])
# S_11 =  np.real(np.abs(vdat[:,-1])**2)
# H_12 = S_12/S_11
# H_i = np.exp(-1j*k*s)
# H_r = np.exp(1j*k*s)

# R_1 = (H_12-H_i)/(H_r-H_12)
# R = R_1*np.exp(1j*2*k*l)
# Z = ((1+R)/(1-R))
# alpha =  1 - abs((Z/(rho*c)-1)/(Z/(rho*c)+1))**2
# alpha = 4* np.real(Z)/((np.real(Z)+1)**2+np.imag(Z)**2)

# fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15)
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].loglog(f,np.real(Z))
# # ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
# ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
# ax[0].set_xlim([100,4e3])
# ax[0].grid()

# ax[1].tick_params(axis = 'x', labelsize=0)
# ax[1].loglog(f,np.imag(Z))
# ax[1].loglog(f,-np.tan(k*.08)**-1)

# # ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
# ax[1].set_xlim([100,4e3])
# ax[1].grid()

# ax[-1].semilogx(f,alpha)
# # ax[-1].set_ylabel('$Phase \ [\circ]$')
# ax[-1].set_xlabel('Frequency [Hz]')
# ax[-1].grid()
# ax[-1].set_xlim([100,4e3])
# ax[-1].set_ylim([0, 1])
# ax[-1].legend(['Waveguide','Basic Acoustic Element','FEM'])

# #%%
# # speed of sound [m/s]
# # c = 343
# # df = 1
# # f = np.arange(1,10e3/df+1)*df
# #   density [kg/m3]

# #%%
# a_n ,a_c,L_n , L_c = 1e-2/2,1e-2/2,1e-10,1e-2
# df = 10
# f = np.arange(1,20e3/df+1)*df

# # a_n ,a_c,L_n , L_c = 0.0254/4,0.0254,0.02,0.05
# helm1 = resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)
# # helm1.set_vdata(vdat_file=vdat_file)

# # f = helm1.vdat[:,0]
# # f0 = c/(4*(L_c+L_c))
# helm1.set_Z(f,WG = True,rad = False,loss = False)
# Z_WG = helm1.Z/(rho*c**2)
# # p_c = helm1.Zb_c/helm1.Z
# # alpha_WG = helm1.alpha


# helm1.set_Z(f,WG = False,rad = False,loss=False)
# Z_BAE = helm1.Z/(rho*c**2)

# # Z_m = 1j*2*np.pi*f*rho*L_c/2*1/(np.pi*a_c**2)
# # Z_c = 1/(1j*2*np.pi*f*(np.pi*a_c**2*L_c/(rho*c**2)))

# # T = np.array([[1+Z_m/Z_c,Z_m],[1/Z_c,np.ones(len(f))]])
# # T_tot = T.transpose(-1,0,1)@[[1],[0]]
# # Z_test_2 = np.squeeze(T_tot[:,0]/T_tot[:,1])
# # Z_test = Z_m+Z_c
# # v = np.squeeze(T_tot[:,1])
# # p_avg = Z_test_2*v
# # alpha_BAE = helm1.alpha


# # P_c = helm1.Z_c/helm1.Z
# # U = helm1.Z**-1

# # f0  = 250
# # helm1.minimize_Z(f0)
# # helm1.set_Z(f,WG = True)
# # Z_WG = helm1.Z
# # alpha_WG = helm1.alpha

# # helm1.set_Z(f,WG = False)
# # Z_BAE = helm1.Z
# # alpha_BAE = helm1.alpha
# #%%

# fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15)
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].loglog(f,np.real(Z_WG))
# ax[0].loglog(f,np.real(Z_BAE))
# # ax[0].loglog(f,np.real(Z))
# # ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
# ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
# ax[0].set_xlim([100,4e3])
# ax[0].grid()

# ax[1].tick_params(axis = 'x', labelsize=0)
# ax[1].loglog(f,np.imag(Z_WG))
# ax[1].loglog(f,np.imag(Z_BAE))
# # ax[1].loglog(f,np.imag(Z))
# # ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
# ax[1].set_xlim([100,4e3])
# ax[1].grid()

# ax[-1].semilogx(f,np.angle(Z_WG)*180/np.pi)
# ax[-1].semilogx(f,np.angle(Z_BAE)*180/np.pi)
# # ax[-1].semilogx(f,np.angle(Z)*180/np.pi)
# ax[-1].set_ylabel('$Phase \ [\circ]$')
# ax[-1].set_xlabel('Frequency [Hz]')
# ax[-1].grid()
# ax[-1].set_xlim([100,4e3])
# ax[-1].set_ylim([-180, 180])
# ax[-1].legend(['Waveguide','Basic Acoustic Element','FEM'])


# #%%
# fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15)
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].loglog(f,np.real(Z_WG**-1))
# ax[0].loglog(f,np.real(Z_BAE**-1))
# # ax[0].loglog(f,np.real(Z**-1))
# # ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
# ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
# ax[0].set_xlim([100,20e3])
# ax[0].grid()

# ax[1].tick_params(axis = 'x', labelsize=0)
# ax[1].loglog(f,np.imag(Z_WG**-1))
# ax[1].loglog(f,np.imag(Z_BAE**-1))
# # ax[1].loglog(f,np.imag(Z**-1))
# # ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
# ax[1].set_xlim([100,20e3])
# ax[1].grid()

# ax[-1].semilogx(f,np.angle(Z_WG**-1)*180/np.pi)
# ax[-1].semilogx(f,np.angle(Z_BAE**-1)*180/np.pi)
# # ax[-1].semilogx(f,np.angle(Z**-1)*180/np.pi)
# ax[-1].set_ylabel('$Phase \ [\circ]$')
# ax[-1].set_xlabel('Frequency [Hz]')
# ax[-1].grid()
# ax[-1].set_xlim([100,20e3])
# ax[-1].set_ylim([-180, 180])
# ax[-1].legend(['Waveguide','Basic Acoustic Element','FEM'])

# #%%
# # fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
# # plt.subplots_adjust(bottom = 0.15)
# # # ax.tick_params(axis = 'x', labelsize=0)
# # # ax.semilogx(helm1.vdat[:,0],helm1.vdat[:,1])
# # # ax.semilogx(f,np.imag(P_c),linestyle = '-.')
# # ax.semilogx(f,np.real(P_c),linestyle = '-.')
# # # ax.semilogx(f,np.imag(P_c),linestyle = '-.')

# # # ax.semilogx(f,np.real(helm1.Z**-1),linestyle = ':')
# # # ax.semilogx(f,np.imag(helm1.Z**-1),linestyle = '-.')
# # # ax.semilogx(f,np.real(helm1.Z**-1)+np.imag(helm1.Z**-1),linestyle = '-.')
# # # ax.semilogx(f,np.real(helm1.Z_c/helm1.Z)+np.imag(helm1.Z_c/helm1.Z),linestyle = '-.')

# # ax.set_ylabel('$P_{cavity} \ [Pa]$')
# # ax.set_xlabel('$Frequency [Hz]$')
# # ax.set_xlim([100,10e3])
# # ax.grid()

# # ax[1].set_xlim([100,10e3])
# # ax[1].tick_params(axis = 'x', labelsize=0)
# # ax[1].loglog(f,np.imag(Z_WG**-1))
# # ax[1].loglog(f,np.imag(Z_BAE**-1))
# # ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# # ax[1].grid()


# #%%

# # fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
# # plt.subplots_adjust(bottom = 0.15)
# # ax.plot(f,abs(np.imag(Z_WG**-1)))
# # ax.plot(f,abs(np.imag(Z_BAE**-1)))
# # # # ax.plot(f,P_c)
# # # ax.plot(f,np.real(P_c))

# # # ax.plot(f,np.real(np.squeeze(T_tot[:,0])))
# # # # ax.plot(f,np.real(1+Z_m/Z_c))
# # # ax.plot(f,np.real((helm1.T@[1,0])[:,0]))

# # # ax.plot(f,np.real(p_avg)**-1)

# # # ax.plot(f,np.imag(np.squeeze(T_tot[:,1])))
# # # ax.plot(f,np.imag(Z_c**-1))
# # # ax.plot(f,np.imag((helm1.T@[1,0])[:,1]))

# # # ax.plot(f,abs(np.imag(Z_BAE**-1)))
# # ax.set_yscale('log')
# # ax.set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# # ax.set_xlim([0,800])
# # ax.set_ylim([10e-8,10e-3])
# # ax.set_xlabel('Frequency [Hz]')
# # ax.grid()
# # ax.legend(['Waveguide','Basic Acoustic Element'])


# # # ax[0].tick_params(axis = 'x', labelsize=0)
# # # ax[0].loglog(f,np.real(Z_WG**-1))
# # # ax[0].loglog(f,np.real(Z_BAE**-1))
# # # ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
# # # ax[0].set_xlim([100,10e3])
# # # ax[0].grid()


# # # ax[1].set_xlim([100,10e3])
# # # ax[1].tick_params(axis = 'x', labelsize=0)
# # # ax[1].loglog(f,np.imag(Z_WG**-1))
# # # ax[1].loglog(f,np.imag(Z_BAE**-1))
# # # ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# # # ax[1].grid()

# # # ax[-1].plot(f,np.real((helm1.T_c[:,0,0]/helm1.T_c[:,1,0])/helm1.Z))
# # # ax[-1].plot(f,np.imag((helm1.T_c[:,0,0]/helm1.T_c[:,1,0])/helm1.Z))
# # # ax[-1].plot(f,np.real((helm1.T_c[:,0,0]/helm1.T_c[:,1,0])/helm1.Z)+np.imag((helm1.T_c[:,0,0]/helm1.T_c[:,1,0])/helm1.Z))

# # # ax[-1].plot(f,np.abs(helm1.T[:,0,0]**-1))
# # # ax[-1].plot(f,alpha_WG)
# # # ax[-1].plot(f,np.angle(((helm1.T_c[:,0,0]/helm1.T_c[:,1,0])/helm1.Z)*180/np.pi))
# # # ax[-1].plot(f,np.abs(U))
# # # ax[-1].set_xscale('log')
# # # ax[-1].set_xlim([100,10e3])
# # # ax[-1].set_ylabel(r'$\alpha$')
# # # ax[-1].set_xlabel('Frequency [Hz]')
# # # ax[-1].grid()
# # # ax[-1].legend(['Waveguide','Basic Acoustic Element'])

# # # ax[-1].set_xlim([100,10e3])
# # # ax[-1].tick_params(axis = 'x', labelsize=0)
# # # ax[-1].plot(w/(2*np.pi),np.abs(Z_c/Z))
# # # ax[-1].set_yscale('log')
# # # ax[-1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# # # ax[-1].grid()

# # # ax[-1].loglog(w/(2*np.pi),np.angle(Z**-1)*180/np.pi)
# # # ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae**-1)*180/np.pi)
# # # ax[-1].plot(w/(2*np.pi),alpha)
# # # ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae**-1)*180/np.pi)

# # # ax[-1].plot(f,alpha_WG)
# # # ax[-1].plot(f,alpha_BAE)

# # ax[-1].set_xscale('log')
# # ax[-1].set_xlim([0,800])
# # ax[-1].set_ylabel(r'$\alpha$')
# # ax[-1].set_xlabel('Frequency [Hz]')
# # ax[-1].grid()
# # ax[-1].legend(['Waveguide','Basic Acoustic Element'])



# # #%%
# # # natural frequency [Hz]
# # f0 = 750
# # w0 = 2*np.pi*f0
# # k0 = w0/c

# # #Airfoil cross-sectional area [m^2] (upper bound of constraint)
# # XsectA = 0.00064214
# # # Blade radius - root cutout [m^2]
# # R = 0.78364

# # A_n,L_n,A_c,L_c = XsectA/2,R/2,XsectA/2,R/2

# # # add constraint for length of neck and cavity = cannot be negative
# # # con = LinearConstraint([[1,0,0,0],[0,0,1,0],[0,1,0,1],[0,1,0,0],[0,0,0,1]],lb = [1e-3*XsectA,1e-3*XsectA,0,1e-7,1e-7],ub = [XsectA,XsectA, R,np.inf ,np.inf  ])
# # con = NonlinearConstraint(fun = con_fun,lb = [0,0,0,0,0,0],ub = [XsectA,XsectA,np.inf,np.inf,R,1])
# # # res = least_squares(opt_wrap,x0 = [A_n,L_n,A_c,L_c],bounds = [0,1],args = [w0])
# # res = minimize(opt_wrap,x0 = [A_n,L_n,A_c,L_c],constraints = con,args = w0,method = 'trust-constr')

# # print(res.x)
# # A_n,L_n,A_c,L_c = res.x
# # R_n = np.sqrt(A_n/np.pi)
# # R_c = np.sqrt(A_c/np.pi)

# # Z = Z_helmholtz(A_n,L_n,A_c,L_c,w,loss=False)
# # Z_bae = Z_helmholtz(A_n,L_n,A_c,L_c,w,wg = False,loss = False)

# # R = (Z-1)/(Z+1)
# # alpha = 1-np.abs(R)**2

# # #%%

# fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15)
# ax[0].tick_params(axis = 'x', labelsize=0)
# ax[0].loglog(w/(2*np.pi),np.real(Z[0]**-1))
# ax[0].loglog(w/(2*np.pi),np.real(Z_bae[0]**-1))
# ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
# ax[0].set_xlim([500,20e3])
# ax[0].grid()

# ax[1].set_xlim([100,20e3])
# ax[1].tick_params(axis = 'x', labelsize=0)
# ax[1].loglog(w/(2*np.pi),np.imag(Z[0]**-1))
# ax[1].loglog(w/(2*np.pi),np.imag(Z_bae[0]**-1))
# ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
# ax[1].grid()

# ax[-1].loglog(w/(2*np.pi),np.angle(Z[0]**-1)*180/np.pi)
# ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae[0]**-1)*180/np.pi)
# ax[-1].set_ylabel('$Phase \ [\circ]$')
# ax[-1].set_xlabel('Frequency [Hz]')
# ax[-1].grid()
# ax[-1].set_xlim([100,320e3])
# ax[-1].set_ylim([-180, 180])
# ax[-1].legend(['Waveguide','Basic Acoustic Element'])

# # ax[-1].set_xlim([100,10e3])
# # ax[-1].loglog(w/(2*np.pi),np.angle(Z[0]**-1)*180/np.pi)
# # ax[-1].loglog(w/(2*np.pi),np.angle(Z_bae[0]**-1)*180/np.pi)
# # ax[-1].set_ylabel('$Phase \ [\circ]$')
# # ax[-1].set_xlabel('Frequency [Hz]')
# # ax[-1].grid()
# # ax[-1].set_xlim([100,10e3])
# # ax[-1].set_ylim([-180, 180])
# # ax[-1].legend(['Waveguide','Basic Acoustic Element'])


# # plt.show()
