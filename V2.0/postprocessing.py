'''
Postprocessing of several simulations with different operating points
Import results from csv files and generate contour plots
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib
import os

import constants
import plot_fun

# define path to csv files
path = 'results_csv\\'
path_plots = 'postprocessing_plots\\'

reduced_R_v_plotting = True
plot_all_phasefrac = True

area = np.pi*(constants.R**2)

x_label = 'Sauter mean diameter / $\mathrm{\mu m}$'
y_label = 'Load / m$^3$ $\cdot$ m$^{-2}$ $\cdot$ h$^{-1}$'
z_label = 'Height of dense-packed zone / mm'
z2_label = 'Phase fraction at outlet / -'
font_size = 30

plt.rcParams['font.sans-serif'] = 'Arial'

# create folder for plots
os.makedirs(path_plots, exist_ok=True)



if reduced_R_v_plotting:
    # filter files that contain "reduced_R_v" in the name and not "secondary"
    files_rv = glob.glob(path + '*reduced_rv*')
    # find csv files that do not contain secondary in the name
    files2 = glob.glob(path + '*secondary*')
    files_rv = [x for x in files_rv if x not in files2]
    # iterate over files and import csv as dataframe
    for i in files_rv:
        # read file into dataframe
        df = pd.read_csv(i, index_col=0)
        # add last row into postprocessed Dataframe df_post as new row
        if i == files_rv[0]:
            df_post_rv = df.iloc[-1,:].to_frame().transpose()
        else:
            df_post_rv = df_post_rv.append(df.iloc[-1,:].to_frame().transpose(), ignore_index=True)
    # calculate height of dense-packed zone
    df_post_rv['h_dp'] = df_post_rv['Dense-packed zone height'] - df_post_rv['Water height']
    # set negative h_dp to 0
    df_post_rv.loc[df_post_rv['h_dp'] < 0, 'h_dp'] = 0
    # safe df_post_rv as csv
    df_post_rv.to_csv(path + 'postprocessed_rv.csv')

    # prepare data as 2D for surface plot
    x = df_post_rv['Sauter mean diameter inlet']
    y = df_post_rv['Aq flow rate inlet']
    # delete duplicates
    x = np.unique(x)
    y = np.unique(y)
    X, Y = np.meshgrid(x, y)
    z = df_post_rv['h_dp']
    z2 = df_post_rv['Hold up at outlet']
    # locate z values at x and y
    Z = np.zeros((len(y), len(x)))
    Z2 = np.zeros((len(y), len(x)))
    for i in range(len(x)):
            for j in range(len(y)):
                # find loc of z value corresponding to x and y
                loc = np.where((df_post_rv['Sauter mean diameter inlet'] == x[i]) & (df_post_rv['Aq flow rate inlet'] == y[j]))
                # add z value to Z with found loc
                Z[j,i] = z.iloc[loc[0][0]]
                Z2[j,i] = z2.iloc[loc[0][0]]

    # plot contour plot filled with color
    # plot reduced_R_v

    fig, ax = plt.subplots(figsize=(10,8))
    CS = ax.contourf(X*1e6, Y/area*3600, Z*1e3, 5, alpha=1, linewidth=2, cmap='viridis')
    ax.clabel(CS, inline=1, fontsize=font_size)
    # add colorbar with label
    cbar = fig.colorbar(CS)
        # increase colorbar font size
    # add label to colorbar
    cbar.ax.set_ylabel('Height of dense-packed zone / mm', size=font_size)
    # increase colorbar font size
    cbar.ax.tick_params(labelsize=font_size)
    ax.set_xlabel(x_label, size=font_size)
    ax.set_ylabel(y_label, size=font_size)
    # increase font size of ticks
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.set_title('Height of dense-packed zone / mm  \n for reduced coalescence parameter by 50%')
    # save plots
    fig.savefig(path_plots + 'h_dp_phasefrac_reduced_rv.png', dpi=1000)
    fig.savefig(path_plots + 'h_dp_phasefrac_reduced_rv.eps', dpi=1000)
    fig.savefig(path_plots + 'h_dp_phasefrac_reduced_rv.svg', dpi=1000)


    # phase fraction outlet as log scale
    fig, ax = plt.subplots(figsize=(10,8))
    CS3 = ax.contourf(X*1e6, Y/area*3600, Z2, 20, alpha=1, linewidth=2, cmap='viridis', norm=matplotlib.colors.LogNorm())
    ax.clabel(CS3, inline=1, fontsize=font_size)
    cbar = fig.colorbar(CS3)
    # add label to colorbar
    cbar.ax.set_ylabel('Phase fraction / -', size=font_size)
    # increase colorbar font size
    cbar.ax.tick_params(labelsize=font_size)
    ax.set_xlabel(x_label, size=font_size)
    ax.set_ylabel(y_label, size=font_size)
    # ax.set_title('Phase fraction at outlet / - \n for reduced coalescence parameter by 50%')
    # increase font size of ticks
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # save plots
    fig.savefig(path_plots + 'phasefrac_outlet_log_reduced_rv.png', dpi=1000)
    fig.savefig(path_plots + 'phasefrac_outlet_log_reduced_rv.eps', dpi=1000)
    fig.savefig(path_plots + 'phasefrac_outlet_log_reduced_rv.svg', dpi=1000)


if plot_all_phasefrac:
    # try to create folder for plots
    try:
        os.mkdir('postprocessing_plots')
    except:
        pass

    # # find csv files that contain secondary in the name
    # files = glob.glob(path + '*secondary*')

    # find csv files that do not contain secondary in the name
    files2 = glob.glob(path + '*secondary*')
    files_all = glob.glob(path + '*.csv')
    files = [x for x in files_all if x not in files2]
    # find csv files that contain 'DN200_L1.8m' in the name
    files = [x for x in files if 'Fine_resolution_Vin_' in x]

    # iterate over files and import csv as dataframe
    for i in files:
        # read file into dataframe
        df = pd.read_csv(i, index_col=0)
        # add last row into postprocessed Dataframe df_post as new row
        if i == files[0]:
            df_post = df.iloc[-1,:].to_frame().transpose()
        else:
            df_post = df_post.append(df.iloc[-1,:].to_frame().transpose(), ignore_index=True)

    # calculate height of dense-packed zone
    df_post['h_dp'] = df_post['Dense-packed zone height'] - df_post['Water height']
    # set negative h_dp to 0
    df_post.loc[df_post['h_dp'] < 0, 'h_dp'] = 0

    # save df_post as csv
    df_post.to_csv(path_plots + 'postprocessed.csv')
    df_post.to_csv(path + 'postprocessed.csv')

    # iterate over all phase fractions 
    phase_frac = np.unique(df_post['Hold up at inlet'])
    for phasefrac in phase_frac:
        # make contour plot of h_dp over Q_w_in and d_32_in for different phase fractions
        # filter dataframe for phase fraction 0.1
        df_post_01 = df_post[df_post['Hold up at inlet'] == phasefrac]

        # prepare data as 2D for surface plot
        x = df_post_01['Sauter mean diameter inlet']
        y = df_post_01['Aq flow rate inlet']
        # delete duplicates
        x = np.unique(x)
        y = np.unique(y)
        X, Y = np.meshgrid(x, y)
        z = df_post_01['h_dp']
        z2 = df_post_01['Hold up at outlet']
        # locate z values at x and y
        Z = np.zeros((len(y), len(x)))
        Z2 = np.zeros((len(y), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                # find loc of z value corresponding to x and y
                loc = np.where((df_post_01['Sauter mean diameter inlet'] == x[i]) & (df_post_01['Aq flow rate inlet'] == y[j]))
                # add z value to Z with found loc
                Z[j,i] = z.iloc[loc[0][0]]
                Z2[j,i] = z2.iloc[loc[0][0]]
                
        # plot contour plot filled with color
        fig, ax = plt.subplots(figsize=(10,8))
        CS = ax.contourf(X*1e6, Y/area*3600, Z*1e3, 5, alpha=1, linewidth=2, cmap='viridis')
        ax.clabel(CS, inline=1, fontsize=font_size)
        # add colorbar with label
        cbar = fig.colorbar(CS)
        # increase colorbar font size
        # add label to colorbar
        cbar.ax.set_ylabel(z_label, size=font_size)
        cbar.ax.tick_params(labelsize=font_size)
        ax.set_xlabel(x_label, size=font_size)
        ax.set_ylabel(y_label, size=font_size)
        # ax.set_title('Height of dense-packed zone / mm  \n at phase fraction' + str(phasefrac) + ' for DN200 L1.8m')
        # increase font size of ticks
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        # save plots
        fig.savefig(path_plots + 'h_dp_phasefrac_' + str(phasefrac) + '.png', dpi=1000)
        fig.savefig(path_plots + 'h_dp_phasefrac_' + str(phasefrac) + '.eps', dpi=1000)
        fig.savefig(path_plots + 'h_dp_phasefrac_' + str(phasefrac) + '.svg', dpi=1000)
        
        fig, ax = plt.subplots(figsize=(10,8))
        # plot contour values in range of 0 to 0.1
        CS2 = ax.contourf(X*1e6, Y/area*3600, Z2, 20, alpha=1, linewidth=2, cmap='viridis')
        ax.clabel(CS2, inline=1, fontsize=font_size)
        # add colorbar with label
        cbar = fig.colorbar(CS2)
        cbar.ax.set_ylabel(z2_label, size=font_size)
        # increase colorbar font size
        cbar.ax.tick_params(labelsize=font_size)
        # add label to colorbar
        cbar.ax.set_ylabel(z2_label, size=font_size)
        ax.set_xlabel(x_label, size=font_size)
        ax.set_ylabel(y_label, size=font_size)
        # ax.set_title('Phase fraction at outlet / - at \n  phase fraction' + str(phasefrac) + ' for DN200 L1.8m')
        # increase font size of ticks
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        # save plots
        fig.savefig(path_plots + 'phasefrac_outlet_phasefrac_' + str(phasefrac) + '.png', dpi=1000)
        fig.savefig(path_plots + 'phasefrac_outlet_phasefrac_' + str(phasefrac) + '.eps', dpi=1000)
        fig.savefig(path_plots + 'phasefrac_outlet_phasefrac_' + str(phasefrac) + '.svg', dpi=1000)
        
        # phase fraction outlet as log scale
        fig, ax = plt.subplots(figsize=(10,8))
        CS3 = ax.contourf(X*1e6, Y/area*3600, Z2, 20, alpha=1, linewidth=2, cmap='viridis', norm=matplotlib.colors.LogNorm())
        ax.clabel(CS3, inline=1, fontsize=font_size)
        cbar = fig.colorbar(CS3)
        cbar.ax.set_ylabel(z2_label, size=font_size)
        # increase colorbar font size
        cbar.ax.tick_params(labelsize=font_size)
        ax.set_xlabel(x_label, size=font_size)
        ax.set_ylabel(y_label, size=font_size)
        # ax.set_title('Phase fraction at outlet / - at \n  phase fraction' + str(phasefrac) + ' for DN200 L1.8m')
        # increase font size of ticks
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        # save plots
        fig.savefig(path_plots + 'phasefrac_outlet_log_phasefrac_' + str(phasefrac) + '.png', dpi=1000)
        fig.savefig(path_plots + 'phasefrac_outlet_log_phasefrac_' + str(phasefrac) + '.eps', dpi=1000)
        fig.savefig(path_plots + 'phasefrac_outlet_log_phasefrac_' + str(phasefrac) + '.svg', dpi=1000)

if 1==0:
    # filter df_post for phase fraction 0.25
    df_post_025 = df_post[df_post['Hold up at inlet'] == 0.25]
    # loop over all q_w_in values and plot h_dp over d_32_in
    for i in np.unique(df_post_025['Aq flow rate inlet']):
        # filter df_post for q_w_in
        df_post_025_1 = df_post_025[df_post_025['Aq flow rate inlet'] == i]
        # plot h_dp over d_32_in
        fig, ax = plt.subplots()
        ax.plot(df_post_025_1['Sauter mean diameter inlet']*1e6, df_post_025_1['h_dp']*1e3, 'o')
        ax.set_xlabel('Sauter mean diameter / um')
        ax.set_ylabel('H_dp / mm')
        ax.set_title('Height of dense-packed zone over Sauter mean diameter \n at load ' + str(round(i/area*3600,1)) + ' m3/m2/h')
        # save plots
        fig.savefig(path_plots + 'h_dp_d32_qw_in_' + str(round(i/area*3600,1)) + '.png', dpi=1000)
        fig.savefig(path_plots + 'h_dp_d32_qw_in_' + str(round(i/area*3600,1)) + '.eps', dpi=1000)
        fig.savefig(path_plots + 'h_dp_d32_qw_in_' + str(round(i/area*3600,1)) + '.svg', dpi=1000)
        
        # plot log hold up at outlet over d_32_in
        fig, ax = plt.subplots()
        ax.plot(df_post_025_1['Sauter mean diameter inlet']*1e6, df_post_025_1['Hold up at outlet'], 'o')
        ax.set_xlabel('Sauter mean diameter / um')
        ax.set_ylabel('Hold up at outlet / -')
        ax.set_yscale('log')
        ax.set_title('Hold up at outlet over Sauter mean diameter \n at load ' + str(round(i/area*3600,1)) + ' m3/m2/h')
        # save plots
        fig.savefig(path_plots + 'hold_up_outlet_d32_qw_in_' + str(round(i/area*3600,1)) + '.png', dpi=1000)
        fig.savefig(path_plots + 'hold_up_outlet_d32_qw_in_' + str(round(i/area*3600,1)) + '.eps', dpi=1000)
        fig.savefig(path_plots + 'hold_up_outlet_d32_qw_in_' + str(round(i/area*3600,1)) + '.svg', dpi=1000)



# # calculate correlation matrix
# corr = df_post.corr()
# # print correlation of h_dp to sauter mean diameter inlet, aq flow rate inlet, phase fraction inlet and hold up at outlet
# print('Correlation of h_dp to sauter mean diameter inlet, aq flow rate inlet, phase fraction inlet and hold up at outlet')
# print(corr['h_dp']['Sauter mean diameter inlet'])
# print(corr['h_dp']['Aq flow rate inlet'])
# print(corr['h_dp']['Hold up at inlet'])
# print(corr['h_dp']['Hold up at outlet'])
# print('Correlation of hold up at outlet to sauter mean diameter inlet, aq flow rate inlet and hold up at inlet')
# print(corr['Hold up at outlet']['Sauter mean diameter inlet'])
# print(corr['Hold up at outlet']['Aq flow rate inlet'])
# print(corr['Hold up at outlet']['Hold up at inlet'])

# # filter df_post for sauter mean diameter inlet greater than 0.4e-3
# df_post_02 = df_post[df_post['Sauter mean diameter inlet'] > 0.4e-3]

# # calculate correlation matrix
# corr = df_post_02.corr()
# # print correlation of h_dp to sauter mean diameter inlet, aq flow rate inlet, phase fraction inlet and hold up at outlet
# print('Correlation for d32_in > 0.4e-3')
# print(corr['h_dp']['Sauter mean diameter inlet'])
# print(corr['h_dp']['Aq flow rate inlet'])
# print(corr['h_dp']['Hold up at inlet'])
# print(corr['h_dp']['Hold up at outlet'])
# print('Correlation of hold up at outlet to sauter mean diameter inlet, aq flow rate inlet and hold up at inlet')
# print(corr['Hold up at outlet']['Sauter mean diameter inlet'])
# print(corr['Hold up at outlet']['Aq flow rate inlet'])
# print(corr['Hold up at outlet']['Hold up at inlet'])

# # filter df_post for water height greater than 0.09 and smaller than 0.11
# df_post_03 = df_post[(df_post['Water height'] > 0.09) & (df_post['Water height'] < 0.11)]
# # calculate correlation matrix
# corr = df_post_03.corr()
# # print correlation of h_dp to sauter mean diameter inlet, aq flow rate inlet, phase fraction inlet and hold up at outlet
# print('Correlation for water height > 0.09 and < 0.11')
# print(corr['h_dp']['Sauter mean diameter inlet'])
# print(corr['h_dp']['Aq flow rate inlet'])
# print(corr['h_dp']['Hold up at inlet'])
# print(corr['h_dp']['Hold up at outlet'])
# print('Correlation of hold up at outlet to sauter mean diameter inlet, aq flow rate inlet and hold up at inlet')
# print(corr['Hold up at outlet']['Sauter mean diameter inlet'])
# print(corr['Hold up at outlet']['Aq flow rate inlet'])
# print(corr['Hold up at outlet']['Hold up at inlet'])

# # plot H_dp over hold up at outlet
# fig, ax = plt.subplots()
# ax.plot(df_post['Hold up at outlet'], df_post['h_dp']*1e3, 'o')
# ax.set_xlabel('Hold up at outlet / -')
# ax.set_ylabel('H_dp / mm')
# ax.set_title('Height of dense-packed zone over hold up at outlet')


# # train PLS model with hold up at outlet, hold-up at inlet, sauter mean diameter inlet and aq flow rate inlet on h_dp
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# # train GP model with hold up at outlet, hold-up at inlet, sauter mean diameter inlet and aq flow rate inlet on h_dp
# from sklearn.gaussian_process import GaussianProcessRegressor

# # generate x and y data
# x = df_post_03[['Hold up at outlet', 'Hold up at inlet', 'Sauter mean diameter inlet', 'Aq flow rate inlet', 'Aq flow rate outlet', 'Organic flow rate outlet', 'Water height']]
# y = df_post_03['h_dp']

# # split data into train and test data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # train PLS model
# pls = PLSRegression(n_components=2)
# pls.fit(x_train, y_train)
# # train GP model
# gp = GaussianProcessRegressor()
# gp.fit(x_train, y_train)

# # predict y values
# y_pred = pls.predict(x_test)
# y_pred_gp = gp.predict(x_test)

# # calculate r2 score
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# print('R2 score of PLS model')
# print(r2)
# print('MSE of PLS model')
# print(mse)

# r2_gp = r2_score(y_test, y_pred_gp)
# mse_gp = mean_squared_error(y_test, y_pred_gp)
# print('R2 score of GP model')
# print(r2_gp)
# print('MSE of GP model')
# print(mse_gp)

# # plot predicted vs. true values
# fig, ax = plt.subplots()
# ax.plot(y_test, y_pred, 'o')
# # add line with slope 1 and +- 20%
# ax.plot(y_test, y_test, 'k')
# ax.plot(y_test, y_test*1.2, 'k--')
# ax.plot(y_test, y_test*0.8, 'k--')
# # add legend
# ax.legend(['Predicted values', 'True values', '20% error margin'])
# ax.set_xlabel('True values')
# ax.set_ylabel('Predicted values')
# ax.set_title('PLS model')

# fig, ax = plt.subplots()
# ax.plot(y_test, y_pred_gp, 'o')
# ax.plot(y_test, y_test, 'k')
# ax.plot(y_test, y_test*1.2, 'k--')
# ax.plot(y_test, y_test*0.8, 'k--')
# ax.legend(['Predicted values', 'True values', '20% error margin'])
# ax.set_xlabel('True values')
# ax.set_ylabel('Predicted values')
# ax.set_title('GP model')







    
    