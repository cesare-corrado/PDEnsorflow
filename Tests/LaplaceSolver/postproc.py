import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def left_IND(IND,dI):
    LL=IND-dI
    return(np.where(LL<0,0,LL))


def right_IND(IND,dI,Imax):
    LL=IND+dI
    return(np.where(LL>Imax,Imax-1,LL))




def savefig(plotdata,Mx,My,figname,minval=None, maxval=None):
    nx,ny,nz = plotdata.shape
    fig=plt.figure(figsize=(100,50))    
    for kk in range(nz):
        ax = fig.add_subplot(My,Mx,1+kk)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.imshow(plotdata[:,:,kk],vmin=minval,vmax=maxval )
    fig.tight_layout() 
    fig.savefig(figname)
    plt.close()



def initialize_envelope(domain,nvox):
    nx,ny,nz = domain.shape
    I,J,K    = np.where(domain.astype(bool))    
    envelope = domain.astype(bool).copy()
    for j in range(nvox):
        dI = j+1
        
        envelope[left_IND(I,dI),J,K]      = True
        envelope[right_IND(I,dI,nx),J,K]  = True
        envelope[I,left_IND(J,dI),K]      = True
        envelope[I,right_IND(J,dI,ny),K ] = True
        envelope[I,J,left_IND(K,dI)]      = True    
        envelope[I,J,right_IND(K,dI,nz)]  = True
        
        if(j==0):
            # cross-voxels on the plane z=K
            envelope[left_IND(I,dI),left_IND(J,dI),K]         = True
            envelope[left_IND(I,dI),right_IND(J,dI,ny),K]     = True
            envelope[right_IND(I,dI,nx),left_IND(J,dI),K]     = True    
            envelope[right_IND(I,dI,nx),right_IND(J,dI,ny),K] = True   
            # Note: for cross in the z direction, use the previous with l/r on Z
            # AND then the L/R on plane for planes Z l/R

            envelope[left_IND(I,dI),left_IND(J,dI),left_IND(K,dI)]         = True
            envelope[left_IND(I,dI),right_IND(J,dI,ny),left_IND(K,dI)]     = True
            envelope[right_IND(I,dI,nx),left_IND(J,dI),left_IND(K,dI)]     = True    
            envelope[right_IND(I,dI,nx),right_IND(J,dI,ny),left_IND(K,dI)] = True   

            envelope[left_IND(I,dI),left_IND(J,dI),right_IND(K,dI,nz)]         = True
            envelope[left_IND(I,dI),right_IND(J,dI,ny),right_IND(K,dI,nz)]     = True
            envelope[right_IND(I,dI,nx),left_IND(J,dI),right_IND(K,dI,nz)]     = True    
            envelope[right_IND(I,dI,nx),right_IND(J,dI,ny),right_IND(K,dI,nz)] = True   

            envelope[left_IND(I,dI),J,left_IND(K,dI)]          = True
            envelope[right_IND(I,dI,nx),J,left_IND(K,dI)]      = True
            envelope[I,left_IND(J,dI),left_IND(K,dI)]          = True
            envelope[I,right_IND(J,dI,ny),left_IND(K,dI) ]     = True
        
            envelope[left_IND(I,dI),J,right_IND(K,dI,nz)]      = True
            envelope[right_IND(I,dI,nx),J,right_IND(K,dI,nz)]  = True
            envelope[I,left_IND(J,dI),right_IND(K,dI,nz)]      = True
            envelope[I,right_IND(J,dI,ny),right_IND(K,dI,nz) ] = True
        
        
        
        
    envelope = envelope.astype(float)
    envelope[domain.astype(bool)] = np.nan
    return(envelope)




def compute_labeling(tdata,domain,nvox,label_dict,thresholds):
    #temperature at time stamp ITMP
    last_temp = np.copy(tdata)                 
    envelope = initialize_envelope(domain,nvox)
    envelope[np.isnan(envelope)] = 1.0  
    last_temp[np.logical_not(envelope.astype(bool))] = np.nan
    #last temp: from [0,1] to {0,10}
    last_temp = np.round(last_temp*10)
    last_temp[domain.astype(bool)]=-1.0  # set domain value to -1

    labeling = last_temp.copy()
    labeling[np.logical_and((last_temp>=thresholds['endo']), (last_temp<=thresholds['epi'] ))] = label_dict['undef_label']
    labeling[last_temp <= thresholds['endo']]                       = label_dict['endo_label']
    labeling[last_temp>=max(thresholds['epi'],thresholds['endo'])]  = label_dict['epi_label']
    labeling[domain.astype(bool)]                                   = label_dict['domain_label']
    return(labeling)



if __name__=='__main__':
    Mx = 16
    My = 8                                                             
    
    nsmpl  = 418  #106
    domain = np.load("domain.npy")
    data   = np.load("cube3D_128_128_128.npy")
    data   = data[:nsmpl,:,:,:] 
    nvox   = 1


    thresholds = {'endo':4, 'epi':7  }
    
    
    label_dict = { 'endo_label':   2.0, #-1.0,
                  'epi_label':     3.0,
                  'domain_label':  1.0,
                  'undef_label':   4.0} 
    
    
    # plot domain envelope (the layer of voxels around the domain = 1.0 and the non-domain voxels =0.0; domain=nan)
    envelope = initialize_envelope(domain,nvox)
    figname  = 'labeling_layers_{0}_vox.png'.format(nvox)
    savefig(envelope,Mx,My,figname)
    
    
    BEST_ITMP = -1
    labeling = compute_labeling(data[0,:,:,:],domain,nvox,label_dict,thresholds)
    nb_endo  = np.count_nonzero(labeling==label_dict['endo_label'])
    nb_epi   = np.count_nonzero(labeling==label_dict['epi_label'])
    adiff = abs(nb_endo-nb_epi)
    
    #Determine the best labeling interval
    for ITMP in range(1,nsmpl):
        labeling = compute_labeling(data[ITMP,:,:,:],domain,nvox,label_dict,thresholds)
        nb_undef = np.count_nonzero(labeling==label_dict['undef_label'])
        nb_endo  = np.count_nonzero(labeling==label_dict['endo_label'])
        nb_epi   = np.count_nonzero(labeling==label_dict['epi_label'])
        nb_tot   = nb_undef + nb_endo + nb_epi
        print('{}; undefined: {:3.3f}; endo: {:3.3f}; epi: {:3.3f} '.format(ITMP,nb_undef/ nb_tot , nb_endo/ nb_tot, nb_epi/ nb_tot )    )
        if abs(nb_endo-nb_epi) <= adiff:
            adiff = abs(nb_endo-nb_epi)
            BEST_ITMP = ITMP
        
        



    
    print('Best time step is {0} with {1}'.format(BEST_ITMP,adiff))     
    #plot temp field
    last_temp = np.copy(data[BEST_ITMP,:,:,:])                 
    figname = 'temp_field_timestamp_{0}.png'.format(BEST_ITMP)
    savefig(last_temp,Mx,My,figname)

    #plot temp field
    envelope = initialize_envelope(domain,nvox)
    envelope[np.isnan(envelope)] = 1.0  #set dmain voxels from nana to 1
    last_temp[np.logical_not(envelope.astype(bool))] = np.nan
    figname = 'temp_labeling_layers_{0}_vox_timestamp_{1}.png'.format(nvox,BEST_ITMP)
    savefig(last_temp,Mx,My,figname,minval=0.0, maxval=1.0)




    labeling = compute_labeling(data[BEST_ITMP,:,:,:],domain,nvox,label_dict,thresholds)    
    nb_undef = np.count_nonzero(labeling==label_dict['undef_label'])
    nb_endo  = np.count_nonzero(labeling==label_dict['endo_label'])
    nb_epi   = np.count_nonzero(labeling==label_dict['epi_label'])
    nb_tot   = nb_undef + nb_endo + nb_epi
    print('{}; undefined: {:3.3f}; endo: {:3.3f}; epi: {:3.3f} '.format(BEST_ITMP,nb_undef/ nb_tot , nb_endo/ nb_tot, nb_epi/ nb_tot )    )
    
    figname = 'labels_{0}_vox_timestamp_{1}.png'.format(nvox,BEST_ITMP)
    savefig(labeling,Mx,My,figname)
    labeling[np.isnan(labeling)]=0.0
    img = nib.Nifti1Image(labeling,np.eye(4) )
    
    segname = 'labels_{0}_vox_timestamp_{1}.nii.gz'.format(nvox,BEST_ITMP)
    nib.save(img, segname)
    
    

