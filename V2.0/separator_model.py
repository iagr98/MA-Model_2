'''
Separator Model based on Backi

'''
import os
from scipy.optimize import minimize

class SeparatorModel:
    
    def __init__(self, physical_properties, geometry_parameter, numerical_parameter, model_parameter, operating_conditions, initial_conditions):
        self.physical_properties = physical_properties
        self.geometry_parameter = geometry_parameter
        self.numerical_parameter = numerical_parameter
        self.model_parameter = model_parameter
        # self.operating_conditions = operating_conditions
        # self.initial_conditions = initial_conditions
        # settings
        self.name_of_simulation = operating_conditions.name_of_simulation
        self.do_plot = operating_conditions.do_plot
        self.t_end = numerical_parameter.t_end
        # path with current date and time
        self.path = 'results' + str("\\") + self.name_of_simulation
        # create folder where plots and logfiles are saved 
        try:
            os.mkdir('results')
        except:
            pass
        try:
            os.mkdir('results_csv')
        except:
            pass
        
        # operating conditions
        self.Q_in = operating_conditions.Q_in
        self.phasefraction_in = operating_conditions.phasefraction_in
        self.d32_in = operating_conditions.Sauterdiameter_in
        self.Q_w_out = operating_conditions.Q_w_out
        self.Q_o_out = operating_conditions.Q_o_out
        
        # physical properties
        self.r_v = physical_properties.r_v
        self.eta_w = physical_properties.eta_w
        self.rho_w = physical_properties.rho_w
        self.rho_o = physical_properties.rho_o
        self.eta_o = physical_properties.eta_o
        self.sigma = physical_properties.sigma
        self.delta_rho = physical_properties.delta_rho
        
        # initial conditions
        self.h_w_initial = initial_conditions.h_w
        self.h_l_initial = initial_conditions.h_l
        self.h_dp_initial = initial_conditions.h_dp
        
        # boundary conditions
        self.n_in, self.d_bins, self.N_in_total = 0, 0, 0
        
        # differential variables
        self.h_w = 0
        self.h_l = 0
        self.h_dp = 0

        # output variables
        self.V_dp = 0
        self.E = 0
        self.L_dp = 0
        
    # update operating conditions
    def update_operating_conditions(self, operating_conditions):
        """Updates operating conditions from a tuple

        Args:
            operating_conditions (tuple): Tuple of operating conditions (Q_in, phasefraction_in, d32_in, Q_w_out, Q_o_out)

        Returns:
            none
        """
        self.Q_in = operating_conditions[0]
        self.phasefraction_in = operating_conditions[1]
        self.d32_in = operating_conditions[2]
        self.Q_w_out = operating_conditions[3]
        self.Q_o_out = operating_conditions[4]
        # print updated operating conditions
        print('Updated operating conditions: ')
        print('Q_in: ', self.Q_in)
        print('phasefraction_in: ', self.phasefraction_in)
        print('d32_in: ', self.d32_in)
        print('Q_w_out: ', self.Q_w_out)
        print('Q_o_out: ', self.Q_o_out)
    
    def update_initial_conditions(self, initial_conditions):
        """Updates initial conditions from a tuple

        Args:
            initial_conditions (tuple): Tuple of initial conditions (h_w, h_l, h_dp)

        Returns:
            none
        """
        self.h_w_initial = initial_conditions[0]
        self.h_l_initial = initial_conditions[1]
        self.h_dp_initial = initial_conditions[2]
        # print updated initial conditions
        print('Updated initial conditions: ')
        print('h_w_initial: ', self.h_w_initial)
        print('h_l_initial: ', self.h_l_initial)
        print('h_dp_initial: ', self.h_dp_initial)
        
    def update_plot_settings(self, do_plot):
        """Updates plot settings

        Args:
            do_plot (bool): True if plots should be created, False otherwise

        Returns:
            none
        """
        self.do_plot = do_plot
        # print updated plot settings
        print('Updated plot settings: ')
        print('do_plot: ', self.do_plot)
    
    def update_name_of_simulation(self, name_of_simulation):
        """Updates name of simulation

        Args:
            name_of_simulation (str): Name of simulation

        Returns:
            none
        """
        self.name_of_simulation = name_of_simulation
        # path with current date and time
        self.path = 'results' + str("\\") + self.name_of_simulation
        # print updated name of simulation
        print('Updated name of simulation: ')
        print('name_of_simulation: ', self.name_of_simulation)
    
    def update_t_end(self, t_end):
        """Updates end time of simulation

        Args:
            t_end (float): End time of simulation

        Returns:
            none
        """
        self.t_end = t_end
        # print updated end time of simulation
        print('Updated end time of simulation: ')
        print('t_end: ', self.t_end)
        
    def update_r_v(self, r_v):
        """Updates r_v

        Args:
            r_v (float): r_v

        Returns:
            none
        """
        self.r_v = r_v
        # print updated r_v
        print('Updated r_v: ')
        print('r_v: ', self.r_v)

    def update_physical_properties_aq(self, physical_properties_aq):
        """Updates physical properties of aqueous phase

        Args:
            physical_properties_aq (tuple): Tuple of physical properties of aqueous phase (eta_w, rho_w)

        Returns:
            none
        """
        self.eta_w = physical_properties_aq[0]
        self.rho_w = physical_properties_aq[1]
        self.delta_rho = abs(self.rho_w - self.rho_o)
        # print updated physical properties of aqueous phase
        print('Updated physical properties of aqueous phase: ')
        print('eta_w: ', self.eta_w)
        print('rho_w: ', self.rho_w)

    def update_physical_properties(self, physical_properties):
        """Updates physical properties

        Args:
            physical_properties (tuple): Tuple of physical properties (rho_w, eta_w, rho_o, eta_o, sigma)
        Returns:
            _type_: _description_
        """
        self.rho_w = physical_properties[0]
        self.eta_w = physical_properties[1]
        self.rho_o = physical_properties[2]
        self.eta_o = physical_properties[3]
        self.sigma = physical_properties[4]
        self.delta_rho = abs(self.rho_w - self.rho_o)
        
        print('Updated physical properties: ')
        print('rho_w: ', self.rho_w)
        print('eta_w: ', self.eta_w)
        print('rho_o: ', self.rho_o)
        print('eta_o: ', self.eta_o)
        print('sigma: ', self.sigma)

    def get_operating_conditions(self):
        """returns operating conditions

        Returns:
            Tuple: Q_in, phasefraction_in, d32_in, Q_w_out, Q_o_out
        """
        return self.Q_in, self.phasefraction_in, self.d32_in, self.Q_w_out, self.Q_o_out
    # functions
    def get_stokes_vel(self,D):
        # stokes velocity calculation
        eta_c = self.eta_w
        v_s = self.physical_properties.g*D**2*self.delta_rho/(18*eta_c)
        return v_s

    def get_swarm_vel(self,D, epsilon):
        '''
        swarm sedimentation Richardson-Zaki with n=2 from Kampwerth 2020
        input: D: droplet diameter in m
        input: Delta_rho: density difference of dispersed and continuous phase in kg/m3
        input: eta_c: continuous phase viscosity in Pa*s
        input: g: gravitational acceleration in m/s2
        input: epsilon: hold up of dispersed phase in 1
        output: v_s: swarm velocity in m/s
        '''
        n = 2
        v_0 = self.get_stokes_vel(D)
        v_s = v_0 * (1 - epsilon)**(self.model_parameter.n - 1)
        return v_s 
    
    def get_tau_x(self,A,q):
        # horizontal residence time calculation of a segment of length L, crosssectional area A and volume flow q
        tau_h = self.geometry_parameter.l*A/q
        return tau_h

    def get_A_x(self,h_w):
        import numpy as np
        r = self.geometry_parameter.r
        eps = self.numerical_parameter.eps
        # crosssectional area (normal vector in x direction) of separator dependent on height of water h_w and radius of separator r
        if h_w >= 2*r - eps:
            return np.pi*r**2
        if h_w <= eps: # return area of segment equal to 1e-3 m height (to avoid division by zero)
            return eps
        return r**2/2*(2*np.arccos((r-h_w)/r) - np.sin(2*np.arccos((r-h_w)/r)))

    def get_A_y(self,h_i): 
        # crosssectional area (normal vector in y direction) of interface dependent on height of interface h_i and radius of separator r for axial segment length dL
        # catch error if h_i is smaller eq. zero (no DGTS in separator)
        import numpy as np
        r = self.geometry_parameter.r
        eps = self.numerical_parameter.eps
        dL = self.geometry_parameter.l/self.numerical_parameter.N_s
        
        root = h_i*(2*r - h_i)
        if root < eps:
            return 2*dL*eps
        else:
            A_i = 2*dL*np.sqrt(root)
        return A_i

    def get_factor_dAx_dh(self,h):
        # derivative factor for crosssectional area (normal vector in x direction) of separator dependent on height h and separator length l
        import numpy as np
        l = self.geometry_parameter.l
        r = self.geometry_parameter.r
        if h >= 2*r or h <= 0:
            return 1e12 # return infinity if h is equal larger than 2*r or smaller than zero
        else:
            return 1/(2*l*(h*(2*r-h))**0.5)

    def get_tau_y(self,h, v_s):
        '''
        vertical residence time of a droplet with swarm velocity v_S for height h
        input: h: height in m
        input: v_S: swarm velocity in m/s
        output: tau_v: vertical residence time in s
        '''
        # vertical residence time of a droplet with diameter D with stokes velocity for height h
        tau_v = h/v_s
        return tau_v

    def get_V_d(self,D):
        # Volume of droplet with diameter m
        import numpy as np
        V_d = np.pi/6*D**3
        return V_d
    
    def get_V_dp(self):
        # get the volume of the densed-packed zone at the end of the simulation
        import numpy as np
        A_A = np.pi * self.geometry_parameter.r**2
        A_h_w = self.get_A_x(self.h_w)
        A_h_l = self.get_A_x((self.h_l-self.h_dp-self.h_w))
        # return (A_A - A_h_w - A_h_l) * self.geometry_parameter.l
        return (A_A - A_h_w - A_h_l) * self.L_dp
    
    def get_efficiency(self, n):
        N_d = self.numerical_parameter.N_d
        V_out = 0
        V_in = 0
        for i in range(N_d):
            V_out = V_out + n[-1,-1,i]*self.get_V_d(self.d_bins[i])
            V_in = V_in + n[-1,0,i]*self.get_V_d(self.d_bins[i])
        return 1 - (V_out / V_in)

    def get_droplet_classes(self, d_32, d_max=3e-3, s=0.32, plot=False):
        '''
        calculate log-normal DSD function from sauter mean diameter d32[m]
        input: d_32: sauter mean diameter of droplets in m
        input: d_max: maximum droplet diameter in m
        input: s: standard deviation of volume-based log-normal distribution (Kraume2004)
        input: plot: if True plot the distribution
        return n_count_rel: relative number-based probability of droplets for each class based on the derived volume-based log normal distribution
        return d_bins_center: bin center of droplet classes

        see Kraume, Influence of Physical Properties on Drop Size Distributions of Stirred Liquid-Liquid Dispersions, 2004 
        and Ye, Effect of temperature on mixing and separation of stirred liquid/liquid dispersions over a wide range of dispersed phase fractions, 2023
        '''
        import numpy as np
        from scipy.stats import lognorm
        from scipy import stats
        import matplotlib.pyplot as plt
        
        N_d = self.numerical_parameter.N_d

        # statistical representative number of droplets for volume distribution
        N_vol = int(1e6)
        # lognorm volume distribution of d/d_32
        dist = lognorm(s) 
        
        # define bin edges (diameter class) equidistantly from numberClasses
        x = np.linspace(0,d_max/d_32,N_d+1)
        
        if plot==True:
            # plot lognorm distribution
            fig, ax = plt.subplots(1, 1)
            ax.plot(x*d_32*1e6,dist.pdf(x))
            ax.set_ylim([0,1.2*max(dist.pdf(x))])
            ax.set_xlim([0,d_max*1e6])
            ax.set_xlabel('$d / \mathrm{\mu m}$')
            ax.set_ylabel('$q_3 / \mathrm{\mu m}^-1$')
            ax.set_title('Volume-based probability density distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                        + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                        + ', \n number of classes=' + str(N_d))
            # save plot
            fig.savefig(self.path + '\lognorm_dist.png', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist.eps', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist.svg', dpi=1000)
            
        # divide sample points into bins hist[0] is the count and hist[1] the edges of bins
        hist = np.histogram(dist.rvs(N_vol, random_state= 1),bins=x, density=False)
        # return middle value of bins boundary values
        d_bins = hist[1]*d_32
        d_bins_center = np.zeros(len(d_bins)-1)
        for i in range(len(d_bins)-1):
            d_bins_center[i] = (d_bins[i]+d_bins[i+1])/2
            
        # transform volume based absolute distribution to number based relative distribution
        v_count_abs = hist[0]
        n_count_abs = np.zeros(len(v_count_abs))
        v_count_rel = np.zeros(len(v_count_abs))
        for i in range(len(v_count_abs)):
            n_count_abs[i] = v_count_abs[i]*6/(np.pi*d_bins_center[i]**3)
            v_count_rel[i] = v_count_abs[i]/sum(v_count_abs)
        # normalize number distribution
        n_count_rel = np.zeros(len(v_count_abs))
        for i in range(len(v_count_abs)):
            n_count_rel[i] = n_count_abs[i]/sum(n_count_abs)
            
        # optional plotting of transformed distribution
        if plot==True:
            fig, ax = plt.subplots(1, 1)
            ax.plot(d_bins_center*1e6,v_count_rel, label='Volume-based')
            ax.plot(d_bins_center*1e6,n_count_rel, label='Number-based')
            # ax.set_ylim([0,1])
            ax.set_xlim([0,d_max*1e6])
            ax.set_xlabel('$d / \mathrm{\mu m}$')
            ax.set_ylabel('$h $')
            ax.set_xlim([0,d_max*1e6])
            ax.set_ylim([0,1.2*max(np.append(v_count_rel, n_count_rel))])
            ax.set_title('Relative distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                        + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                        + ', \n number of classes=' + str(N_d))
            ax.legend()
            # save plot
            fig.savefig(self.path + '\lognorm_dist_rel.png', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist_rel.eps', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist_rel.svg', dpi=1000)
            
            # plot histogram of number distribution
            fig, ax = plt.subplots(1, 1)
            ax.bar(d_bins_center*1e6,n_count_rel, width=0.5*d_bins_center[1]*1e6)
            ax.set_xlabel('$d / \mathrm{\mu m}$')
            ax.set_ylabel('$h $')
            ax.set_title('Relative number-based distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                        + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                        + ', \n number of classes=' + str(N_d))
            fig.savefig(self.path + '\lognorm_dist_rel_n.png', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist_rel_n.eps', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist_rel_n.svg', dpi=1000)
            # plot histogram of volume distribution
                    # plot histogram of number distribution
            fig, ax = plt.subplots(1, 1)
            ax.bar(d_bins_center*1e6,v_count_rel, width=0.5*d_bins_center[1]*1e6)
            ax.set_xlabel('$d / \mathrm{\mu m}$')
            ax.set_ylabel('$h $')
            ax.set_title('Relative volume-based distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                        + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                        + ', \n number of classes=' + str(N_d))
            fig.savefig(self.path + '\lognorm_dist_rel_v.png', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist_rel_v.eps', dpi=1000)
            fig.savefig(self.path + '\lognorm_dist_rel_v.svg', dpi=1000)

        return n_count_rel, d_bins_center

    def get_totalNumber_water_inlet(self, hold_up, d_32, d_max, V_mix, report=False):
        '''
        calculates the total number of droplets entering the separator for a given hold up and volume of mixing that follows the volume-based lognormal distribution (Kraume2004)
        input: hold_up: hold up of org. in aq. phase entering the separator in 1
        input: d_32: sauter mean diameter of droplets in m
        input: d_max: maximum droplet diameter in m
        input: V_mix: Volume of mixer (volume of first aq. phase segment) in m3
        input: report: if True print results
        output: N_in_total: total number of droplets entering the separator in 1
        '''
        # use minimize to calculate number of droplets
        from scipy import optimize
        N_d = self.numerical_parameter.N_d

        N_in_total = 1e4 # initial guess
        
        # relative number distribution at inlet 
        n_count_rel, d_bins = self.get_droplet_classes(d_32, d_max=d_max)
        
        def f(N_in_total):

            # volume of dispered phase in m3
            V_disp = 0
            for i in range(N_d):
                V_disp = V_disp + N_in_total*n_count_rel[i]*self.get_V_d(d_bins[i])
            # hold up of water in separator
            hold_up_calc = V_disp/V_mix
            return hold_up_calc-hold_up
        
        N_in_total = optimize.newton(f,N_in_total,rtol=1e-4)
            
        # converet number of droplets to integer
        N_in_total = int(N_in_total)
        # calculate hold up for found number of droplets
        hold_up_calc = f(N_in_total) + hold_up
        # print results
        # print hold_up_calc with 4 digits
        hold_up_calc = round(hold_up_calc,4)
        if report:
            print('hold up: ' + str(hold_up_calc))
            print('number of droplets: ' + str(N_in_total))
        return N_in_total, n_count_rel, d_bins

    def get_sauter_mean_diameter_stirrer(self, We):
        '''
        calculates the sauter mean diameter of droplets in m based on the weber number
        input: We=n**2*D**3*rho_disp/sigma: weber number in 1
        output: d_32: sauter mean diameter of droplets in m
        see Kraume 2004 et al.
        '''
        import numpy as np
        d_stirrer = self.geometry_parameter.d_stirrer
        c_2 = 1 # constant depending on stirrer geometry
        n = 0.6 # for breakage dominant mixing processes
        d_32 = d_stirrer*c_2*(We)**(n)
        return d_32

    def get_sauter_mean_diameter(self, n_count_abs, d_bins):
        '''
        calculates the sauter mean diameter of droplets 
        input: n_count_abs: absolute number-based probability of droplets for each class in 1
        input: d_bins_center: bin center of droplet classes in m
        output: d_32: sauter mean diameter of droplets in m
        '''
        import numpy as np
        if np.sum(n_count_abs) != 0:
            v = 0
            o = 0
            for i in range(len(n_count_abs)):
                v = v + n_count_abs[i]*d_bins[i]**3
                o = o + n_count_abs[i]*d_bins[i]**2
            return v/o
        else:
            return 0

    def get_coalescence_time(self, d_32, h_p, at_interface=True):
        '''
        calculates the coalescence time of droplets in s
        input: d_32: sauter mean diameter of droplets in m
        input: h_p: height of the dense-packed zone in m
        input: r_v: asymetric film drainage parameter
        input: Delta_rho: density difference of dispersed and continuous phase in kg/m3
        input: sigma: interfacial tension between dispersed and continuous phase in N/m
        input: eta_c: continuous phase viscosity in Pa*s
        input: g: gravitational acceleration in m/s2
        input: Ha: Hamaker constant in J
        input: at_interface: if True coalescence at interface, if False coalescence at droplet-droplet contact
        output: tau: coalescence time of droplets in s
        '''
        import numpy as np
        r_v = self.r_v
        delta_rho = self.delta_rho
        sigma = self.sigma
        eta_c = self.eta_w
        g = self.physical_properties.g
        Ha = self.physical_properties.Ha
        
        # check if droplets are at interface or droplet-droplet contact
        if d_32 <= 0:
            return np.inf
        La = (delta_rho*g/sigma)**0.6 *h_p**0.2 *d_32
        root = np.sqrt(1 - 4.7 / (La+4.7))
        r_fc = 0.3025 * d_32 * root # contact area radius droplet to droplet
        r_fi = np.sqrt(3) * r_fc # contact area radius droplet to interface
        r_a  = 0.5 * d_32 * (1 - root) # radius of the channel contour
        if at_interface:
            r_f = r_fi
        else:
            r_f = r_fc
        tau = ((6*np.pi)**(7/6) * eta_c * (r_a)**(7/3)) / \
                (4 * sigma**(5/6) * Ha**(1/6) * r_f * r_v) 
        return tau

    def initialize_boundary_conditions(self, epsilon_in, d_32, d_max, plot=False):
        '''
        calculates the boundary conditions for the separator model of Backi et al. 2018
        input: epsilon_in: hold up of water in inlet in 1
        input: d_32: sauter mean diameter of droplets in m
        input: d_max: maximum droplet diameter in m
        output: hold_up_calc: hold up of water in separator in 1
        output: n_in: number of droplets in each class at inlet
        output: d_bins: bin center of droplet classes in m
        output: N_in_total: total number of droplets entering the separator in 1
        '''
        r = self.geometry_parameter.r
        l = self.geometry_parameter.l
        N_s = self.numerical_parameter.N_s
        N_d = self.numerical_parameter.N_d
        
        V_mix = self.get_A_x(r)*l/N_s
        self.N_in_total, n_in_rel, self.d_bins = self.get_totalNumber_water_inlet(epsilon_in,d_32,d_max=d_max, V_mix=V_mix) # total number of droplets at inlet
        # relative dropsize distribution at inlet
        self.get_droplet_classes(d_32, d_max=d_max, plot=plot)
        # convert to absolute number of droplets at inlet
        self.n_in = n_in_rel*self.N_in_total
        # inlet flow rates
        # volume of dispered phase in m3
        V_disp = 0
        for i in range(N_d):
            V_disp = V_disp + self.n_in[i]*self.get_V_d(self.d_bins[i])
        # hold up of water in separator
        self.hold_up_calc = V_disp/V_mix
        return 
    def initialize_input_variables(self):
        self.u = (self.Q_in, self.Q_w_out, self.Q_o_out, self.n_in, self.phasefraction_in)
        self.p = (self.d_bins,)
        return
    def initialize_initial_conditions(self):
        self.y_0 = (self.h_w_initial, self.h_l_initial, self.h_dp_initial)
        # check if initial conditions are physical
        if self.y_0[1] > 2*self.geometry_parameter.r:
            # raise ValueError('Initial liquid height is larger than Separator diameter!')
            print('Initial liquid height is larger than Separator diameter!')
            return
        if self.y_0[2] > self.y_0[1] or self.y_0[2] < self.y_0[0]:
            # raise ValueError('Initial dense-packed zone height is larger than liquid height or smaller than water height!')
            print('Initial dense-packed zone height is larger than liquid height or smaller than water height!')
            return
        if self.y_0[0] < 0:
            # raise ValueError('Initial water height is negative!')
            print('Initial water height is negative!')
            return
        return

    #%% main separator model
    def calculate_separator(self,y,u,p):
        '''
        calculates the separator model of Backi et al. 2018
        input: y: state variables
        input: u: input variables
        input: p: parameter variables
        output: dy_dt_RHS: right hand side of ODE
        output: q_w: water flow rate in each segment
        output: epsilon_w: hold up of water in each segment
        output: q_o: organic flow rate in each segment
        output: q_dp: dispersed phase flow rate in each segment
        output: n: number of droplets in each segment and class
        output: pos: position of droplets in each segment and class
        output: d32_dp: sauter mean diameter of droplets in each segment
        output: tau_di: coalescence time of droplets in each segment
        output: dV_c: volume flow rate of coalescence in each segment
        output: dV_w_dp: volume flow rate of water leaving segment due to coalescence and build up of dense-packed zone
        output: dV_s: volume flow rate of sedimentation in each segment
        output: dV_si: volume flow rate of sedimentation in each segment
        output: dn_dp: number of droplets leaving segment from dense-packed zone
        '''
        import numpy as np
        
        N_s = self.numerical_parameter.N_s
        N_d = self.numerical_parameter.N_d
        pos_in = self.model_parameter.pos_in

        eps = self.numerical_parameter.eps
        epsilon_di = self.model_parameter.epsilon_di
        epsilon_dp = self.model_parameter.epsilon_dp
          
        # assign diff variables
        h_w = y[0]
        h_l = y[1]
        h_dp = y[2]
        
        # assign input variables
        q_w_in = u[0]
        q_w_out = u[1]
        q_o_out = u[2]
        n_in = u[3]
        epsilon_in = u[4]
        D = p[0]

        dy_dt_RHS = np.zeros(3)
        # declaration of variables for each segment and droplet class
        tau_x = np.zeros(N_s)
        tau_y = np.zeros((N_s, N_d))
        dV_s = np.zeros((N_s, N_d))
        dV_w = np.zeros((N_s, N_d))
        n_to_dp = np.zeros((N_s, N_d))
        v_s = np.zeros((N_s, N_d)) # calculated from previous hold-up
        n_dpz = np.zeros((N_s, N_d)) 
        # declaration of variables for each segment
        d32_dp = np.zeros(N_s)
        tau_di = np.zeros(N_s)
        dV_c = np.zeros(N_s)
        dV_w_dp = np.zeros(N_s) 
        dV_si = np.zeros(N_s)
        d32_aq = np.zeros(N_s)
        # declaration of variables entering a segment / N_S + 1 is outlet
        q_w = np.zeros(N_s + 1)
        epsilon_w = np.zeros(N_s + 1)
        q_w_o = np.zeros(N_s + 1)
        q_o = np.zeros(N_s + 1)
        q_dp = np.zeros(N_s + 1)
        # declaration of variables entering a segment and for each class
        n = np.zeros((N_s + 1, N_d))
        pos = np.zeros((N_s + 1, N_d))
        # declaration of variables for each droplet class
        V_d = np.zeros(N_d)
    
        # droplet specific properties
        V_d = self.get_V_d(D)
        
        # boundary conditions (inlet)
        q_w[0] = q_w_in
        epsilon_w[0] = epsilon_in
        q_o[0] = 0
        q_dp[0] = 0
        n[0,:] = n_in
        pos[0,:] = pos_in
        
        A_w = self.get_A_x(h_w) # cross sectional area of heavy phase in x direction
        A_y = self.get_A_y(h_dp) # area of active interface dense-packed zone and coherent light phase
        
        # calculation of rate terms for each segment
        for i in range(N_s):
            ## sedimentation for every segment
            # residence times
            tau_x[i] = self.get_tau_x(A_w,q_w[i])
            v_s[i,:] = self.get_swarm_vel(D,epsilon_w[i])
            tau_y[i,:] = self.get_tau_y(h_w-pos[i,:],v_s[i,:])
            
            #calculate available stream of dispersed phase in aq.
            q_w_o[i] = q_w[i]*epsilon_w[i] 
                
            # for every droplet class
            for k in range(N_d):           
                # partial sedimentation of class k
                if tau_x[i] < tau_y[i,k]:
                    # droplet arrive at dense-packed zone
                    n_to_dp[i,k] = n[i,k]*(tau_x[i]*v_s[i,k]/(h_w - pos[i,k]))
                    # update next segment
                    n[i+1,k] = n[i,k]*(1 - tau_x[i]*v_s[i,k]/(h_w - pos[i,k]))
                    # position update
                    if pos[i,k] + v_s[i,k]*tau_x[i] > h_w:
                        pos[i+1,k] = h_w
                    else:   
                        pos[i+1,k] = pos[i,k] + v_s[i,k]*tau_x[i]
                # full sedimentation of class k
                else:
                    n_to_dp[i,k] = n[i,k]
                    n[i+1,k] = 0
                    pos[i+1,k] = y[0] - eps
                # calculate rate of sedimentation for ODE 
                if np.sum(n[i,:]) != 0:
                    dV_s[i,k] = q_w_o[i]* ((n[i,k] - n[i+1,k])/np.sum(n[i,:]))  
                else:
                    dV_s[i,k] = 0
            # calculate rate of sedimentation for this segment & rate of DSD to dense-packed zone
            dV_si[i] = np.sum(dV_s[i,:])
            # sauter mean diameter of droplets sedimenting to dense-packed zone
            d32_aq[i] = self.get_sauter_mean_diameter(n_to_dp[i,:], D[:])
            
            ## dense-packed zone
            # distribution of droplets in this segment by combining droplets from previous segment and sedimenting droplets
            if i == 0:
                # calculation by absolute numbers
                n_dpz[i,:] = n_to_dp[i,:]
            else:
                # assume mixing of previous segment distribution in dense-packed zone and sedimentating droplets
                if q_dp[i] != 0:
                    n_dpz[i,:] = n_dpz[i-1,:] + n_to_dp[i,:]
                else:
                    n_dpz[i,:] = n_to_dp[i,:]
            d32_dp[i] = self.get_sauter_mean_diameter(n_dpz[i,:], D)   
            # coalescence rate calculation
            if h_dp > h_w + d32_dp[i]/2:
                tau_di[i] = self.get_coalescence_time(d32_dp[i], h_dp-h_w)
            else:
                tau_di[i] = np.inf
            dV_c[i] = 2*d32_dp[i]*A_y*epsilon_di / (3*tau_di[i])
            
            # calculation of remaining flows
            # calculation of pure water stream from heavy phase due to coalescence and build up of dense-packed zone
            dV_w_dp[i] = (dV_si[i] - dV_c[i]) * (1 / epsilon_dp - 1)

            ## update of convective volume flows
            q_o[i+1] = q_o[i] + dV_c[i]
            q_w[i+1] = np.max([q_w[i] - dV_si[i] - dV_w_dp[i], 0])
            q_dp[i+1] = np.max([q_dp[i] + dV_si[i] + dV_w_dp[i] - dV_c[i], 0])

            # update hold-up in aq. phase by volume balance of dispersed droplets in aq. phase
            epsilon_w[i+1] = (epsilon_w[i]*q_w[i] - dV_si[i]) / q_w[i+1]

        # check if dense-packed zone is empty
        if h_dp < h_w + eps:
            dV_c_tot = 0
            dV_w_dp_tot = np.sum(dV_s) * (1 / epsilon_dp - 1)
        else:
            dV_c_tot = np.sum(dV_c)
            dV_w_dp_tot = np.sum(dV_w_dp)
        dV_s_tot = np.sum(dV_s) 
        
        # assigning dy_dt for each segment
        dy_dt_RHS[0] = (q_w_in - q_w_out - dV_s_tot - dV_w_dp_tot) * self.get_factor_dAx_dh(h_w)
        dy_dt_RHS[1] = (q_w_in - q_o_out - q_w_out)* self.get_factor_dAx_dh(h_l)
        dy_dt_RHS[2] = (q_w_in - q_w_out - dV_c_tot)* self.get_factor_dAx_dh(h_dp)
        
        # calculate q_w_out if h_w is constant
        q_w_out_h_w_const = q_w_in - dV_s_tot - dV_w_dp_tot
        # calculate q_o_out if h_w is constant & h_l is constants
        q_o_out_hw_hl_constant = q_w_in - q_w_out_h_w_const
        # calculate q_w_in if h_dp & h_w is constant
        q_w_in_h_dp_const = q_w_out_h_w_const + dV_c_tot
        
        # pack results, flow rates and number distributions into tuple
        results = (dy_dt_RHS, q_w, epsilon_w, q_o, q_dp, n, pos, d32_dp, tau_di, dV_c, dV_w_dp, dV_s, dV_si, n_dpz, q_w_out_h_w_const, q_w_in_h_dp_const, q_o_out_hw_hl_constant)
        
        return results

    def dy_dt_RHS(self, t, y, u, p):
        dy_dt_RHS = self.calculate_separator(y,u,p)[0]
        return dy_dt_RHS
    
    def find_flooding_flowrate(self, h_dp_max):
        '''
        find flooding flow rate of dispersion in m3/s so that h_dp reaches h_dp_max
        input: h_dp_max: maximum height of dense-packed zone in m
        '''
        try:
            os.mkdir(self.path)
        except:
            pass
        # initialize boundary conditions with constant h_dp_max and constant water height
        y_0 = (self.h_w_initial, self.h_l_initial, h_dp_max)
        self.initialize_boundary_conditions(self.phasefraction_in, self.d32_in, d_max=2.5*self.d32_in, plot=self.do_plot) 
        p = (self.d_bins,)
        def residual(Q_in):
            # define input variables
            u = (Q_in, self.Q_w_out, self.Q_o_out, self.n_in, self.phasefraction_in)
            # calculate separator model with constant h_w and h_dp
            q_w_in_h_dp_const = self.calculate_separator(y_0,u,p)[15]
            # return residual
            res = q_w_in_h_dp_const - Q_in
            return res**2
        # res = minimize(residual, self.Q_in, bounds=((self.Q_in/5, self.Q_in*5),), method='Nelder-Mead', tol=1e-10, options={'disp': True})
        res = minimize(residual, self.Q_in, bounds=((self.Q_in/5, self.Q_in*5),), method='L-BFGS-B', tol=1e-11, options={'disp': True})
        self.Q_in_flood = res.x[0]

    #%% ODE solving
    def solve_ODE(self, h_w_const=False, h_l_const=True, print_logfile=True, report=True, calc_algebraic=True):
        '''
        solves the ODE system of the separator model
        input: h_w_const: if True the water height is kept constant by overwriting the aq outflow
        input: h_l_const: if True the liquid height is kept constant by overwriting the org outflow
        input: print_logfile: if True the logfile is printed
        input: report: if True the logfile is printed and terminal output is printed
        input: calc_algebraic: if True the algebraic equations are calculated
        '''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.integrate import solve_ivp
        import os

        import sys
        from datetime import datetime
        
        try:
            os.mkdir(self.path)
        except:
            pass
        
        # initial conditions & parameter
        self.initialize_initial_conditions()
        

        # track computation time
        start_time = datetime.now()

        if print_logfile:
            # log file with current date and time from subfolder
            log_file = open(self.path + '\logfile.log',"w")
            sys.stdout = log_file
            
        # initializing simulation
        if report:
            print('----------------------------------------')  
            print('Separator model calculation')
            print('----------------------------------------')  

        # simulation settings
        # t_end = 100 # end time in s
        time_span = [0, self.t_end] # time span in s

        # boundary conditions / inlet conditions
        d_max = 2.5*self.d32_in # maximum diameter of droplets in m
        # initialize boundary conditions
        self.initialize_boundary_conditions(self.phasefraction_in, self.d32_in, d_max, plot=self.do_plot) 
        # initialize input variables from boundary conditions
        self.initialize_input_variables()

        # print constants with units
        if report:
            print('Constants: number of segments in separator' , self.numerical_parameter.N_s, '-')
            print('Constants: number of droplet classes' , self.numerical_parameter.N_d, '-')
            print('Constants: total flow rate inlet' , self.Q_in*3600, 'm3/h')
            print('Constants: phase fraction at inlet' , self.phasefraction_in, '-')
            print('Constants: radius of separator' , self.geometry_parameter.r, 'm')
            print('Constants: length of separator' , self.geometry_parameter.l, 'm')
            print('Constants: density of organic phase' , self.rho_o, 'kg/m3')
            print('Constants: density of water phase' , self.rho_w, 'kg/m3')
            print('Constants: viscosity of organic phase' , self.eta_o, 'Pa*s')
            print('Constants: viscosity of water phase' , self.eta_w, 'Pa*s')
            print('Constants: density difference' , self.delta_rho, 'kg/m3')
            print('Constants: interfacial tension' , self.sigma, 'N/m')
            print('Constants: asymetric film drainage parameter' , self.r_v, '-')
            print('----------------------------------------')
            # print initial conditions y_0
            print('Initial conditions water height: ', self.y_0[0], 'm')
            print('Initial conditions liquid height: ', self.y_0[1], 'm')
            print('Initial conditions dense-packed zone: ', self.y_0[2], 'm')
            print('----------------------------------------')
            # print boundary conditions with units
            print('Boundary conditions sauter mean diameter: ', self.d32_in*1e6, 'um')
            print('Boundary conditions maximum diameter: ', d_max*1e6, 'um')
            # print boundary conditions with units
            print('Boundary conditions total flow rate liquid inlet: ', (self.Q_in)*3600, 'm3/h')
            print('Boundary conditions aq. phase flow rate inlet: ', self.Q_in*3600, 'm3/h')
            print('Boundary conditions aq. phase flow rate outlet: ', self.Q_w_out*3600, 'm3/h')
            print('Boundary conditions org. phase flow rate outlet: ', self.Q_o_out*3600, 'm3/h')
            print('Boundary conditions holdup in aq. phase: ', self.hold_up_calc, '-')
            print('Boundary conditions droplet classes: ', self.d_bins*1e6, 'um')
            print('Boundary conditions total number of droplets: ', self.N_in_total, '-')
            print('Boundary conditions number of droplets: ', self.n_in, '-')

        # if h_w is constant 
        if h_w_const:
            # find Q_w_out so that ODE for h_w is zero
            Q_w_out_h_w_const = self.calculate_separator(self.y_0, self.u, self.p)[14]
            self.u = (self.Q_in, Q_w_out_h_w_const, self.Q_o_out, self.n_in, self.phasefraction_in)
            # if h_w and h_l is constant
            if h_l_const:
                # find Q_o_out so that ODE for h_l is zero
                Q_o_out_hw_hl_constant = self.calculate_separator(self.y_0, self.u, self.p)[16]
                self.u = (self.Q_in, Q_w_out_h_w_const, Q_o_out_hw_hl_constant, self.n_in, self.phasefraction_in)
                if report:
                    print('Overwriting boundary conditions org. phase flow rate outlet for h_w=const.: ', Q_o_out_hw_hl_constant*3600, 'm3/h')
            if report:
                print('Overwriting boundary conditions aq. phase flow rate outlet for constant h_w & h_w: ', Q_w_out_h_w_const*3600, 'm3/h')   
        
        ## generate inputs for solve_ivp
        inputs = (self.u, self.p)
        solver_options = {'rtol': 1e-12, 'atol': 1e-12}
        
        # define events representing unphysical states
        def flooded_densepackedzone(t, y, u, p):
            # flooded separator with dense-packed phase
            return y[2] + 1e-3 - y[1]
        flooded_densepackedzone.terminal = True
        def flooded_separator(t, y, u, p):
            # flooded separator with liquid phase
            return y[1] - 2*self.geometry_parameter.r - 1e-3
        flooded_separator.terminal = True
        def empty_aq_phase(t, y, u, p):
            # empty separator with aq. phase
            return y[0] - 1e-3
        empty_aq_phase.terminal = True

        events_physics = (flooded_densepackedzone, flooded_separator, empty_aq_phase)

        if report:
            print('Solving ivp...')
        sol = solve_ivp(self.dy_dt_RHS, time_span, self.y_0, args=inputs, dense_output=True, \
            method='RK45', events=events_physics, options=solver_options, t_eval=np.linspace(0, self.t_end, 10*self.t_end+1))
        if report:
            print(sol)

        # plot solution of solve_ivp up to first occuring event
        if sol.status == 0:
            t_range = np.linspace(0, sol.t[-1], int(10*sol.t[-1]+1))
        if sol.status == 1: # evenet occured  
            # check which t_events array is not empty
            for i in range(len(sol.t_events)):
                if sol.t_events[i].size != 0:
                    t_range = np.linspace(0, sol.t_events[i][0], int(10*sol.t_events[i][0]+1))
                    self.t_end = sol.t_events[i][0]
        if sol.status == -1: # integration failed and terminate script
            if report:
                print('Integration failed!')
            return
            
        z = sol.sol(t_range)
        if self.do_plot:
            fig, ax = plt.subplots()
            ax.plot(t_range, z.T[:,0]*1e3, label='Water')   
            ax.plot(t_range, z.T[:,1]*1e3, label='Liquid')
            ax.plot(t_range, z.T[:,2]*1e3, label='Dense-packed zone')
            ax.set_ylim([0,2.1*self.geometry_parameter.r*1e3])
            ax.set_xlim([0, self.t_end])
            ax.set_xlabel('Time / s')
            ax.set_ylabel('Liquid heights / mm')
            ax.set_title('Two-phase l-l separator DN' + str(2*self.geometry_parameter.r*1e3) + ', L=' + str(self.geometry_parameter.l) + 'm')
            ax.legend()
            # save plot
            fig.savefig(self.path + '\liquid_heights.png', dpi=1000)
            fig.savefig(self.path + '\liquid_heights.eps', dpi=1000)
            fig.savefig(self.path + '\liquid_heights.svg', dpi=1000)
        
        # assign results to variables
        self.h_w = z.T[-1,0]
        self.h_l = z.T[-1,1]
        self.h_dp = z.T[-1,2]
        
        # dimension of time vector
        N_t = len(t_range)
        
        if calc_algebraic:     
            # numerical parameters
            N_s = self.numerical_parameter.N_s
            N_d = self.numerical_parameter.N_d
            if report:
                # outputs
                print(sol.t_events)

                print('----------------------------------------')
                print('Start calculation of algebraic variables')
                print('----------------------------------------')
            # declaration of algebraic variables
            dy_dt = np.zeros((N_t, 3))
            # coalescence rate
            dV_s = np.zeros((N_t, N_s, N_d))
            dV_si = np.zeros((N_t, N_s))
            dV_s_tot = np.zeros(N_t)
            epsilon = np.zeros((N_t, N_s + 1))
            dV_c = np.zeros((N_t, N_s))
            d32_dp = np.zeros((N_t, N_s))
            d32_out = np.zeros(N_t)
            dV_w_dp = np.zeros((N_t, N_s))
            tau = np.zeros((N_t, N_s))
            q_w = np.zeros((N_t, N_s+1))
            q_dp = np.zeros((N_t, N_s+1))
            q_w_out_hw_const = np.zeros(N_t)
            q_o_out_hw_hl_const = np.zeros(N_t)
            # declaration of variables entering a segment and for each class
            n = np.zeros((N_t, N_s + 1, N_d))
            pos = np.zeros((N_t, N_s + 1, N_d))

            # calculation of algebraic variables
            # loop over time steps
            for t in range(N_t):
                self.results = self.calculate_separator(z.T[t,:], self.u, self.p)
                # allocate results
                # dy_dt[t,:] = results[0]
                # q_w[t,:] = results[1]
                epsilon[t,:] = self.results[2]
                q_dp[t,:] = self.results[4]
                n[t,:,:] = self.results[5]
                # pos[t,:,:] = results[6]
                d32_dp[t,:] = self.results[7]
                # tau[t,:] = results[8]
                dV_c[t,:] = self.results[9]
                dV_w_dp[t,:] = self.results[10]
                # dV_s[t,:,:] = results[11]
                dV_si[t,:] = self.results[12]
                dV_s_tot[t] = np.sum(dV_si[t,:])
                q_w_out_hw_const = self.results[14]
                q_o_out_hl_const = self.results[16]
                d32_out[t] = self.get_sauter_mean_diameter(n[t,-1,:], self.d_bins)
            dV_w_dp_tot = np.sum(dV_w_dp, axis=1)

            L_dp = self.geometry_parameter.l 
            # find length where q_dp is zero
            for i in range(N_s):
                    if q_dp[-1,1+i] == 0:
                        L_dp = i*self.geometry_parameter.l/N_s
                        self.L_dp = L_dp
                        if report:
                            print('q_dp is zero at length: ', L_dp, 'm')
                        break

        ## assigning output variables
        self.V_dp = self.get_V_dp()
        self.E = self.get_efficiency(n)

        # print height and volume of dense-packed zone amd efficiency of the separator at end of simulation
        if report:
            print('Height of dense-packed zone at end of simulation: ', (z.T[-1,2]-z.T[-1,0])*1e3, 'mm.', ' Volume: ', self.V_dp, 'm^3')
            print('Efficieny of separetor at the end of simulation: ', self.E*100, '[%]')

        ## save results as dataframe and export 
        df = pd.DataFrame(data=z.T, columns=['Water height', 'Liquid height', 'Dense-packed zone height']) # 
        # add time vector at first column
        df.insert(0, 'Time', t_range)
        if calc_algebraic:
            df['Coalescence rate'] = np.sum(dV_c, axis=1)
            df['Water feed to dense-packed zone'] = dV_w_dp_tot
            df['Hold up at outlet'] = epsilon[:,-1]
            df['Sedimentation rate'] = dV_s_tot
            df['Sauter mean diameter outlet'] = d32_out
        # add control inputs
        df['Aq flow rate inlet'] = self.Q_in*np.ones(N_t)
        df['Aq flow rate outlet'] = self.Q_w_out*np.ones(N_t)
        df['Organic flow rate outlet'] = self.Q_o_out*np.ones(N_t)
        df['Hold up at inlet'] = self.phasefraction_in*np.ones(N_t)
        df['Sauter mean diameter inlet'] = self.d32_in*np.ones(N_t)
        df['Coalescence parameter'] = self.r_v*np.ones(N_t)
        df['Aq flow rate outlet h_w const'] = q_w_out_hw_const*np.ones(N_t)
        df['Organic flow rate outlet h_l const'] = q_o_out_hl_const*np.ones(N_t)
        df.to_csv(self.path + '\\results.csv')
        #save in separate folder for csv files
        path_csv = datetime.now().strftime('results_csv' + str("\\") + '%Y_%m_%d_%H_%M_' + self.name_of_simulation)
        df.to_csv(path_csv+'.csv')

        ## plotting
        if self.do_plot:
            self.plot_holdup_outlet(t_range, epsilon, self.t_end, self.path)
            self.plot_number_distribution_outlet(t_range, n, self.d_bins, d_max, self.t_end, self.path)
            self.plot_flowrates(t_range, dV_c, dV_s_tot, dV_w_dp, self.t_end, self.path)
            self.plot_d32_over_time(t_range, d32_dp, d_max, self.t_end, self.path)
            self.plot_holdup_over_length(t_range, epsilon, self.t_end, self.path)
            self.plot_qdp(q_dp = q_dp, t_range = t_range, path = self.path)

        # track computation time
        end_time = datetime.now()
        if report:
            print('----------------------------------------')
            print('Computation time: ', end_time - start_time)
            print('----------------------------------------')
        
        ## save results: L_dp, height of dense-packed zone, sol.status, t_end as dataframe and export
        df2 = pd.DataFrame() 
        if calc_algebraic:
            df2['L_dp'] = [L_dp]    
        df2['Height of dense-packed zone'] = [(z.T[-1,2]-z.T[-1,0])]
        df2['sol.status'] = [sol.status]
        df2['t_end'] = [self.t_end]
        df2['CompTime'] = [end_time - start_time]
        df2.to_csv(self.path + '\\results_secondary.csv')
        df2.to_csv(path_csv+'_secondary.csv')

        if print_logfile:
            # print log outputs
            sys.stdout = sys.__stdout__
            log_file.close()
    

    #%% plotting functions 

    
    def plot_holdup_outlet(self,t_range, epsilon, t_end, path):
        '''
        plot hold up in aq. phase outlet
        '''
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot(t_range, epsilon[:,-1], label='Aq. phase hold up outlet')
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Hold up / -')
        ax.set_xlim([0, t_end])
        ax.set_ylim([0, 1.1*max(epsilon[:,-1])])
        ax.legend()
        # save plot
        fig.savefig(path+'/hold_up_outlet.png', dpi=1000)
        fig.savefig(path+'/hold_up_outlet.eps', dpi=1000)
        fig.savefig(path+'/hold_up_outlet.svg', dpi=1000)

    def plot_number_distribution_outlet(self, t_range, n, d_bins, d_max, t_end, path):
        '''
        plot number distribution at outlet for several time steps
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        # get time id at every 20% of t_end
        t_id = np.linspace(0, t_end, 6)
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[1].plot(d_bins*1e6, n[0, 0,:]/np.sum(n[0,0,:]), label='Inlet', color='blue')
        for i in range(len(t_id)):
            t_i = int(t_id[i])
            if np.sum(n[t_i,-1,:]) == 0:
                print('No droplets at outlet at t=t_end!')
            else:
                # axs[1].bar(d_bins*1e6, n[t_i,-1,:]/np.sum(n[t_i,-1,:]), label='Outlet',width=0.5*d_bins[1]*1e6)
                axs[1].plot(d_bins*1e6, n[t_i,-1,:]/np.sum(n[t_i,0,:]), label='t=' + str(round(t_id[i],2)) + 's')
        axs[1].set_ylabel('Relative number of droplets')
        axs[1].set_xlim([0, d_max*1e6])
        axs[1].set_ylim([0, max(n[-1, 0,:]/np.sum(n[-1, 0,:]))*1.1])
        axs[1].legend()
        axs[0].bar(d_bins*1e6, n[-1, 0,:]/np.sum(n[-1,0,:]), label='Inlet',width=0.5*d_bins[1]*1e6)
        axs[0].plot(d_bins*1e6, n[-1, 0,:]/np.sum(n[-1,0,:]), color='blue')
        axs[0].set_xlabel('Droplet diameter / um')
        axs[0].set_ylabel('Relative number of droplets')
        axs[0].set_xlim([0, d_max*1e6])
        axs[0].set_ylim([0, max(n[-1,0,:]/np.sum(n[-1,0,:]))*1.1])
        axs[0].set_title('Relative number distribution at t=t_end')
        axs[0].legend()
        # save plot
        fig.savefig(path + '\\number_distribution_outlet_tend.png', dpi=1000)
        fig.savefig(path + '\\number_distribution_outlet_tend.eps', dpi=1000)
        fig.savefig(path + '\\number_distribution_outlet_tend.svg', dpi=1000)
        
        
    def plot_flowrates(self,t_range, dV_c, dV_s_tot, dV_w_dp, t_end, path):
        '''
        plot coalescence rate and flow rates over time
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        ax.plot(t_range, np.sum(dV_c, axis=1)*3600*1e3, label='Coalescence rate')
        ax.plot(t_range, dV_s_tot[:]*3600*1e3, label='Separation rate')
        ax.plot(t_range, np.sum(dV_w_dp,axis=1)*3600*1e3, label='Water feed to dense-packed zone')
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Flow rates / L/h')
        ax.set_xlim([0, t_end])
        # ax.set_ylim([0, 2])
        ax.legend()
        # save plot
        fig.savefig(path + '\\flow_rates.png', dpi=1000)
        fig.savefig(path + '\\flow_rates.eps', dpi=1000)
        fig.savefig(path + '\\flow_rates.svg', dpi=1000)

    def plot_d32_over_time(self, t_range, d32_dp, d_max, t_end, path):
        '''
        plot d32 over length with time labels
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        
        l = self.geometry_parameter.l
        N_s = self.numerical_parameter.N_s
        
        
        # plot d32 over segments/length for different times with time labels
        x = np.arange(N_s+1)*l/(N_s)
        # delete first element of x
        x = np.delete(x, 0)
        # get time id at every 20% of t_end
        t_id = np.linspace(0, t_end, 6)

        fig, ax = plt.subplots()
        for i in range(len(t_id)):
            t_i = int(t_id[i])
            ax.plot(x, d32_dp[t_i,:]*1e6, label='t=' + str(round(t_id[i],2)) + 's')
        ax.set_xlabel('Length / m')
        ax.set_ylabel('d32 / um')
        ax.set_xlim([0, x[-1]])
        ax.set_ylim([0, d_max*1e6])
        ax.set_title('Sauter mean diameter in dense-packed zone')
        ax.legend()
        # save plot
        fig.savefig(path + '\\d32.png', dpi=1000)
        fig.savefig(path + '\\d32.eps', dpi=1000)
        fig.savefig(path + '\\d32.svg', dpi=1000)
        
    def plot_holdup_over_length(self, t_range, epsilon, t_end, path):
        '''
        plot holdup in aq. phase over length with time labels
        '''
        import matplotlib.pyplot as plt
        import numpy as np
                
        l = self.geometry_parameter.l
        N_s = self.numerical_parameter.N_s
        
        # plot d32 over segments/length for different times with time labels
        x = np.arange(N_s+1)*l/(N_s)
        # get time id at every 20% of t_end
        t_id = np.linspace(0, t_end, 6)
        fig, ax = plt.subplots()
        for i in range(len(t_id)):
            t_i = int(t_id[i])
            ax.plot(x, epsilon[t_i,:], label='t=' + str(round(t_id[i],2)) + 's')
        ax.set_xlabel('Length / m')
        ax.set_ylabel('Hold-up / -')
        ax.set_xlim([0, x[-1]])
        ax.set_title('Hold-up in aq. phase')
        ax.legend()
        # save plot
        fig.savefig(path + '\\hold_up_aq_Phase.png', dpi=1000)
        fig.savefig(path + '\\hold_up_aq_Phase.eps', dpi=1000)
        fig.savefig(path + '\\hold_up_aq_Phase.svg', dpi=1000)

    def plot_qdp(self, q_dp, t_range, path):
        '''
        make 3D plot of convective flow rates over segments and time
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        
        l = self.geometry_parameter.l
        N_s = self.numerical_parameter.N_s
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.arange(N_s+1)*l/(N_s)
        # delete first element of x
        y = t_range
        X, Y = np.meshgrid(x, y)
        Z = q_dp*3600
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
        ax.set_xlabel('Length / m')
        ax.set_ylabel('Time / s')
        ax.set_zlabel('Flow rate / m3/h')
        # save plot
        fig.savefig(path + '\\convective_flow_rates_3D.png', dpi=1000)
        fig.savefig(path + '\\convective_flow_rates_3D.eps', dpi=1000)
        fig.savefig(path + '\\convective_flow_rates_3D.svg', dpi=1000)

