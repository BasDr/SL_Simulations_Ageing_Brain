import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

filepath_in= 'Location of empirical FC'
filepath_out= 'Location where saved'

indices_label_order = [0, 2, 4, 6, 8, 12, 14, 16, 20, 22, 24, 26, 28, 30, 34, 32, 36, 38, 40, 42, 44, 46, 
                       48, 50, 52, 54, 56, 58, 60, 62, 10, 64, 66, 18, 1, 3, 5, 7, 9, 13, 15, 17, 21, 23, 
                       25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 11, 
                       65, 67, 19]

def Calculate_mean_FC():
    youth=['CC110037', 'CC110182', 'CC120376', 'CC120137', 'CC120061', 'CC120550', 'CC110606', 'CC120212', 'CC121685', 'CC120347', 'CC110056', 'CC110126', 'CC110098', 'CC110101', 'CC120276', 'CC120727', 'CC122016', 'CC110033', 'CC110045', 'CC110187', 'CC110411', 'CC120065', 'CC120120', 'CC120182', 'CC120640', 'CC121144', 'CC120218', 'CC120309', 'CC110069', 'CC110087', 'CC120049', 'CC120264', 'CC112141', 'CC210519', 'CC210148', 'CC210174', 'CC210023', 'CC210172', 'CC220098', 'CC220511', 'CC220535', 'CC220223', 'CC220323', 'CC221595', 'CC210314', 'CC210422', 'CC210617', 'CC220107', 'CC220203', 'CC221336']
    elder=['CC510086', 'CC510355', 'CC510237', 'CC510304', 'CC510226', 'CC520134', 'CC520279', 'CC610099', 'CC610288', 'CC610496', 'CC610625', 'CC620118', 'CC620499', 'CC610178', 'CC610292', 'CC610469', 'CC620557', 'CC620567', 'CC610210', 'CC610392', 'CC610462', 'CC610568', 'CC610052', 'CC610076', 'CC610658', 'CC620354', 'CC621284', 'CC710223', 'CC720103', 'CC720238', 'CC710350', 'CC720622', 'CC720023', 'CC720071', 'CC720407', 'CC710088', 'CC710154', 'CC710342', 'CC720290', 'CC720516', 'CC710131', 'CC710551', 'CC710591', 'CC720400', 'CC721374', 'CC723395', 'CC712027', 'CC720774', 'CC721224', 'CC711035']
    FC_samples_youth=[]
    FC_samples_elder=[]
    for file in range(len(youth)):
        y_data= np.load(os.path.join(filepath_in, f'{youth[file]}_aec.npy')) #adjust to how the FC are called
        e_data= np.load(os.path.join(filepath_in, f'{elder[file]}_aec.npy')) #adjust to how the FC are called

        y_data=y_data[np.ix_(indices_label_order,indices_label_order)]
        e_data=e_data[np.ix_(indices_label_order,indices_label_order)]
        

        #y_data= np.mean(y_data, axis=0)
        #e_data = np.mean(e_data, axis=0)
        
        FC_samples_youth.append(y_data)
        FC_samples_elder.append(e_data)

    mean_FC_y= np.mean(FC_samples_youth, axis=0)
    mean_FC_e= np.mean(FC_samples_elder, axis=0)
    np.fill_diagonal(mean_FC_y, 0)
    np.fill_diagonal(mean_FC_e, 0)

    np.save(os.path.join(filepath_out, 'MeanFC_youth'), mean_FC_y)
    np.save(os.path.join(filepath_out, 'MeanFC_elder'), mean_FC_e)

    plt.figure(figsize=(16,12))
    cc = plt.imshow(mean_FC_y, cmap='viridis', clim=np.percentile(mean_FC_y, [5, 95]), aspect='auto')
    plt.title('Mean FC youth')
    plt.xlabel('Regions')
    plt.ylabel('Regions')
    #plt.savefig('MeanFC_youth_new')
    #plt.savefig('MeanFC_youth')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(cc, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(filepath_out,'MeanFC_youth_mne-package'))

    plt.figure(figsize=(16,12))
    cc= plt.imshow(mean_FC_e, cmap='viridis', clim=np.percentile(mean_FC_e, [5, 95]), aspect='auto')
    plt.title('Mean FC elderly')
    plt.xlabel('Regions')
    plt.ylabel('Regions')
    #plt.savefig('MeanFC_elder_new')
    #plt.savefig('MeanFC_elder')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(cc, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(filepath_out,'MeanFC_elder_mne-package'))

Calculate_mean_FC()
