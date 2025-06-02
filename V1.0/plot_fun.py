'''
plotting functions
'''

def plot_holdup_outlet(t_range, epsilon, t_end, path):
    '''
    plot hold up in aq. phase outlet
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
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

def plot_number_distribution_outlet(t_range, n, d_bins, d_max, t_end, path):
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
    
    
def plot_flowrates(t_range, dV_c, dV_s_tot, dV_w_dp, t_end, path):
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

def plot_d32_over_time(t_range, d32_dp, d_max, t_end, path):
    '''
    plot d32 over length with time labels
    '''
    import constants
    import matplotlib.pyplot as plt
    import numpy as np
    # plot d32 over segments/length for different times with time labels
    x = np.arange(constants.N_S+1)*constants.L/(constants.N_S)
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
    
def plot_holdup_over_length(t_range, epsilon, t_end, path):
    '''
    plot holdup in aq. phase over length with time labels
    '''
    import constants
    import matplotlib.pyplot as plt
    import numpy as np
    # plot d32 over segments/length for different times with time labels
    x = np.arange(constants.N_S+1)*constants.L/(constants.N_S)
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

def plot_qdp(q_dp, t_range, path):
    '''
    make 3D plot of convective flow rates over segments and time
    '''
    import constants
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.arange(constants.N_S+1)*constants.L/(constants.N_S)
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