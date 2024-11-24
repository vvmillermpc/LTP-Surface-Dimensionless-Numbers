import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import sympy as sym
from sympy import Matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os
import torch
import scipy
import matplotlib.pyplot as plt
import platform



def top_split_y(X, Y, percent):
    # Calculate the number of samples for test set
    a = 0
    if isinstance(X, torch.Tensor):
        X = X.numpy()
        a = 1
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    
    print('flag 1')
    print('Y.shape  =', Y.shape)
    print('X.shape  =', Y.shape)
    test_size = int(percent/100 * len(Y))
    print('test_size = ', test_size)
    # Sort Y and get indices of the top 15% values
    sorted_indices = np.argsort(Y[:,0])[::-1][:test_size].flatten()
    print('sorted indices = ', sorted_indices)
    # Create test set
    X_test = X[sorted_indices,:]
    Y_test = Y[sorted_indices]
    print('flag 2')
    print('Y_test.shape  = ', Y_test.shape)
    print('X_test.shape  =', X_test.shape)
    # Create train set
    X_train = np.delete(X, sorted_indices, axis=0)
    Y_train = np.delete(Y, sorted_indices, axis=0)
    print('flag 3')
    print('Y_train.shape  =', Y_train.shape)
    print('X_train.shape  =', X_train.shape)
    if a ==1:
        X_test = torch.tensor(X_test)
        Y_test = torch.tensor(Y_test)
        X_train = torch.tensor(X_train)
        Y_train = torch.tensor(Y_train)
    print('flag 4')
    print('Y_test.shape  =', Y_test.shape)
    print('X_test.shape  =', X_test.shape)
    print('Y_train.shape  =', Y_train.shape)
    print('X_train.shape  =', X_train.shape)
    return X_test, Y_test, X_train, Y_train

def top_split_x(X,Y,percent, index):
# Assuming X is your input array and Y is your output vector


    # Get indices of X[:, 3] sorted in descending order
    sorted_indices = np.argsort(X[:, index])[::-1]

    # Calculate the number of samples for test set
    test_size = int(percent/100 * len(Y))

    # Get the indices for the test set
    test_indices = sorted_indices[:test_size]

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Filter out the test set
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    # Filter out the training set
    X_train = np.delete(X, test_indices, axis=0)
    Y_train = np.delete(Y, test_indices, axis=0)
    return X_test, Y_test, X_train, Y_train


import os
import pandas as pd
import numpy as np

def process_excel_files(folder_path, sheet_name = 'ext dimension number'):
    results = {}
    
    #what will be stored in results in each iteration...
    # Example usage
    # folder_path = r'C:\Users\vvmil\Documents\Python_Vmil\Jupyter_Notebooks\Plasma_He_calcs\He_code_new\dim_num_eb_val_test\unpacked\6_terms'
    # data = process_excel_files(folder_path)
    
    lam_list = []
    
    top_test = {'val_r2' : [],
                'test_r2' : [],
                'card_w' : []
               }
    
    top_val = {'val_r2': [],
               'test_r2':[],
               'card_w':[]
              }
    
    sub_top_val ={'val_r2': [],
               'test_r2':[],
               'card_w':[]
              }
    
    top_5 = {'val_r2': [],
               'test_r2':[],
               'card_w':[]
              }
    
    converged = {'val_r2': [],
               'test_r2':[],
               'card_w':[]
              }
    
    frac = {'frac': [],
               'card_w':[]
            }
    
    
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"The path '{folder_path}' is not a valid directory.")
        return results

    # Iterate through each folder in the directory
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            # Look for Excel files in the subfolder
            
                       
            for file in os.listdir(subfolder_path):
                if file.endswith('.xlsx'):
                    excel_path = os.path.join(subfolder_path, file)
                    # Read the Excel file and access the desired sheet
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)
                    # Store values from specific columns
                    #for index, row in df.iterrows():
                    card_w = df['cardinality_of_w_dim1'].to_numpy()
                    val = df['r2'].to_numpy()
                    test = df['r2_ext_test'].to_numpy()
                    lam = df['lambda'].to_numpy()
                    
                    lam_list.append(np.max(lam))
                    
                    #for top test score
                    a = np.argmax(test)
                    top_test['val_r2'].append(val[a])
                    top_test['test_r2'].append(test[a])
                    top_test['card_w'].append(card_w[a])
                    
                    #for top validation scorer
                    a = np.argmax(val)
                    top_val['val_r2'].append(val[a])
                    top_val['test_r2'].append(test[a])
                    top_val['card_w'].append(card_w[a])
                    
                    # for substitutes
                    # we want the top TEST value out of the validation scores that exceed 0.999.
                    
                    threshold = np.max(val)*0.99
                    b = val>=threshold
                    a = np.argmax(test*b) #the only test values that are multiplied by 1 are where the val > threshold
                    
                    # if we failed to get an index where the validation is above the threshold, 
                    # then a will return zero. In that case, just give me the top performer for now.
                    if a ==0 and val[a]<threshold:
                        a = np.argmax(val)
                        sub_top_val['val_r2'].append(val[a])
                        sub_top_val['test_r2'].append(test[a])
                        sub_top_val['card_w'].append(card_w[a])
                    else:
                        # note we keep a value what it was beofre the ii
                        sub_top_val['val_r2'].append(val[a])
                        sub_top_val['test_r2'].append(test[a])
                        sub_top_val['card_w'].append(card_w[a])
                        
                        
                    
                    # for top 5 val scores
                    a = np.argsort(val)[-4:] # we sort it from smallest to largest, take last 5
                    top_5['val_r2'].append(np.mean(val[a]))
                    top_5['test_r2'].append(np.mean(test[a]))
                    top_5['card_w'].append(np.mean(card_w[a]))
                    
                    
                    # for converged
                    threshold = 0.9
                    a = val>=threshold
                    print('number of val over threshold is', a)
                    if np.sum(a) > 0:
                        converged['val_r2'].append(np.mean(val[a]))
                        converged['test_r2'].append(np.mean(test[a]))
                        converged['card_w'].append(np.mean(card_w[a]))
                    else:
                        converged['val_r2'].append(0)
                        converged['test_r2'].append(0)
                        converged['card_w'].append(11)
                        
                    # fractional: what frac of dim #s that surpassed threshold had test nums that surpassed threshold?
                    threshold = 0.9
                    good_val = val>= threshold
                    count_good_val = np.sum(good_val)
                    good_test = test>=threshold
                    good_val_test = good_test*good_val #we want all of the val numbers > threshold that have tests > threshold
                    count_good_val_test = np.sum(good_val_test)
                    if count_good_val > 0 and np.sum(good_val_test)>0:
                        fraction = count_good_val_test/count_good_val
                        frac['frac'].append(fraction)
                        frac['card_w'].append(np.mean(card_w[good_val_test]))
                    else:
                        fraction = 0
                        frac['frac'].append(fraction)
                        frac['card_w'].append(11)
                    
    results = {'lambda' : lam_list,
            'top_test' : top_test,
           'top_val' : top_val,
            'sub_top_val' : sub_top_val,
            'top_5'  : top_5,
            'converged' : converged,
            'frac': frac
           }

    return results


def plot_dict_vs_list(data_dict, x_list, y_titles, x_titles, sup_titles):
    
    # Example Usage:
    
    # folder_path = r'C:\Users\vvmil\Documents\Python_Vmil\Jupyter_Notebooks\Plasma_He_calcs\He_code_new\dim_num_eb_val_test\unpacked\6_terms'
    # data = process_excel_files(folder_path)
    
    # plot_dict_vs_list(data['top_test'], data['lambda'], y_titles, x_titles, r'Dim Num from Top Test Score')
    # plot_dict_vs_list(data['top_val'], data['lambda'], y_titles, x_titles, r'Dim Num from Top Val Score')
    # plot_dict_vs_list(data['sub_top_val'], data['lambda'], y_titles, x_titles, r'Dim Num from Best Test of Top Val Scores')
    # plot_dict_vs_list(data['top_5'], data['lambda'], y_titles, x_titles, r'Dim Num from Top 5 Val Score')
    # plot_dict_vs_list(data['converged'], data['lambda'], y_titles, x_titles, r'Dim Num from Converged Val Scores')
    # plot_dict_vs_list(data['frac'], data['lambda'], ['hit rate', r"$|\boldsymbol{w}_{\Pi}|_0$"], [r"$\lambda_1$",r"$\lambda_1$"], r'Dim Num Where Val > 0.9, Test > 0.9'  )



    if len(x_list) != len(next(iter(data_dict.values()))):
        print("Error: The length of the list must match the length of the dictionary's vectors.")
        return
    
    import sys
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    sys.path.append("Users\vvmil\AppData\Local\Programs\MiKTeX")
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'
    #%matplotlib inline
    #%config InlineBackend.figure_format='retina'
    
    num_keys = len(data_dict)
    
    fig, axes = plt.subplots(1, num_keys, sharey=False, figsize=(8, 4))
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}
    matplotlib.rc('font', **font)

    i = 0
    plot_list = [r'Test ', r'Extrapolation ', r'']
    for key, y_vector in data_dict.items():
        
        y_vector = np.array(y_vector)
        axes[i].semilogx(x_list,np.where(y_vector > 0*y_vector,y_vector, 0*y_vector) , 'r-o', label = r"R$^2$ vs $\lambda_{1}$")
        axes[i].set_xlabel(x_titles[i], fontsize = 17)

        axes[i].set_ylabel(plot_list[i]+y_titles[i], fontsize = 17)
        #axes[i].set_title(key)
        axes[i].tick_params(axis='both', labelsize=17)
        # Enable minor ticks
        axes[i].minorticks_on()
        # Customize minor ticks (optional)
        axes[i].tick_params(axis='x', which='minor', length=3, width=1) 

        # Set minor tick locator for logarithmic scale
        minor_locator = LogLocator(base=10.0, subs=np.linspace(2, 10, num=9), numticks=10)
        axes[i].xaxis.set_minor_locator(minor_locator)

        i = i+1
    fig.tight_layout(pad=0.8)
    fig.suptitle(sup_titles, **font)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the subplots to make room for the suptitle

    return fig


def get_rate_tuning(data_dict):
    
    rates = []
    
    top_val = {'val_r2' : [],
               'test_r2' :[],
            'card_w' : []
                }
    
    top_5 = {'val_r2': [],
             'test_r2' : [],
            'card_w':[]
              }
    
    
    converged = {'val_r2' : [],
                 'test_r2' : [],
            'card_w' : []
                }
    
    
    for key, file_path in data_dict.items():
        # Convert the key to an integer and append it to the list
        rates.append(int(key))
        # Load the Excel file into a DataFrame
        df = pd.read_excel(file_path , sheet_name = 'ext dimension number')

        # Find the maximum value in the first column and append it to the list
        val = df['r2'].to_numpy() #test
        test = df['r2_ext_test'].to_numpy() #extrapolation
        card_w = df['cardinality_of_w_dim1'].to_numpy()
        
        
        # store the top val score
        a = np.argmax(val)
        top_val['val_r2'].append(np.max([0.0,val[a]]))
        top_val['test_r2'].append(np.max([0.0,test[a]]))
        top_val['card_w'].append(np.max([0.0,card_w[a]])) # not sure why i put np.max here. won't do anything.

        
        # store the average of the converged scores
                # for top 5 val scores
        a = np.argsort(val)[-4:] # we sort it from smallest to largest, take last 5
        top_5['val_r2'].append(np.max([0.0,np.mean(val[a])]))
        top_5['test_r2'].append(np.max([0.0,np.mean(test[a])]))
        top_5['card_w'].append(np.max([0.0,np.mean(card_w[a])]))


        # for converged
        threshold = 0.0
        a = val>=threshold
        if np.sum(a) > 0:
            converged['val_r2'].append(np.mean(val[a]))
            converged['test_r2'].append(np.mean(test[a]))
            converged['card_w'].append(np.mean(card_w[a]))
        else:
            converged['val_r2'].append(0)
            converged['test_r2'].append(np.nan)
            converged['card_w'].append(np.nan)

        results = {'top_val' : top_val,
                    'top_5' : top_5,
                   'converged': converged
                    }
            
    return rates, results


def multi_axes_plot(rates, terms, best_r2, best_r2_card):
    import numpy as np
    import matplotlib
    import sys
    sys.path.append("Users\vvmil\AppData\Local\Programs\MiKTeX")
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'
    import matplotlib.pyplot as plt


    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=False, figsize=(8, 4))


    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 16}

    matplotlib.rc('font', **font)

    ax1.plot(terms,best_r2, 'r-o', label = r"R$^2$ vs $\lambda_{gamma}$")
    ax1.set_xlabel(r"Number of Process Variables", fontsize = 16)
    ax1.set_ylabel(r"Test $R^2$", fontsize = 16)
    x_min = np.min(terms)
    x_max = np.max(terms)
    ax1.set_xlim(x_min-1,x_max+1)
    ax1.set_xticks(range(x_min-1, x_max+2)[0:-1:2])


    ax1_2 = ax1.twiny()
    ax1_2.xaxis.set_ticks_position('bottom')  # Set the position of the second x-axis
    ax1_2.xaxis.set_label_position('bottom')  # Set the position of the second x-axis label
    ax1_2.spines['bottom'].set_position(('outward', 60))  # Offset the second x-axis
    ax1_2.set_xlabel('Number of Largest Rates')
    ax1_2.set_xticks(terms)
    #ax1_2.set_xticklabels([str(i) for i in rates]+['All'])
    ax1_2.set_xticklabels([str(i) for i in rates])

    ax1_2.set_xlim(x_min-1,x_max+1)



    ax2.plot(terms,best_r2_card, '-o', label = r"$||\gamma||_0$ vs $\lambda_{gamma}$")

    ax2.set_ylabel(r"$||\boldsymbol{w}_1||_0$", fontsize = 16)
    ax2.set_xlabel(r"Number of Process Variables", fontsize = 16)
    ax2.set_xlim(x_min-1,x_max+1)
    ax2.set_xticks(range(x_min-1, x_max+2)[0:-1:2])

    ax2_2 = ax2.twiny()
    ax2_2.xaxis.set_ticks_position('bottom')  # Set the position of the second x-axis
    ax2_2.xaxis.set_label_position('bottom')  # Set the position of the second x-axis label
    ax2_2.spines['bottom'].set_position(('outward', 60))  # Offset the second x-axis
    ax2_2.set_xlabel('Number of Largest Rates')
    ax2_2.set_xticks(terms)
    #ax2_2.set_xticklabels([str(i) for i in rates]+['All'])
    ax2_2.set_xticklabels([str(i) for i in rates])
    ax2_2.set_xlim(x_min-1,x_max+1)


    fig.tight_layout(pad=0.8)

    #ax1.yaxis.labelpad=10.0
    #fig.suptitle(r'Dimensionless Regression with Mass Balance Base Terms', fontsize=16, y=.95)

    return fig
