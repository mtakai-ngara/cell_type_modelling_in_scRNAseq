import cell_discovery_limits_utils_test as utils
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams


def visualize_mod_genes(norm_fname,fc,no_genes,no_cells,output_dir,modification_method='multiplicative',
                  gene_modification_profile_df=None,quantiles=None,preselect_quantile=None,
                  preselected_genes=None,genes_in_all_cells=False,nTopExpressed=4500,topVarGenes=500):
    if experiment_name: 
        output_dir=utils.os.path.join(output_dir,experiment_name)
        utils.safe_mkdir(output_dir)            
    if gene_modification_profile_df:        
        gene_modification_profile_df=gene_modification_profile_df.astype({'genes':str})
        gene_modification_profile_df = gene_modification_profile_df[(gene_modification_profile_df.genes!='nan') &
                                                                    (gene_modification_profile_df.fc.isin([fc]))]
        gene_modification_profile_df=None if gene_modification_profile_df.empty else gene_modification_profile_df
    norm_indata=read_inputdata(filepath=norm_fname,sep='\t',header=0,index_col=0)    
    if nTopExpressed:        
        norm_indata_mean_series=norm_indata.mean(axis=1)        
        top_expr_gns=pd.Series.sort_values(norm_indata_mean_series,ascending=False).iloc[0:nTopExpressed].index.tolist()        
        preselected_genes = top_expr_gns if preselected_genes is None else list(set(preselected_genes).intersection(top_expr_gns))         
    if preselect_quantile:
        preselected_quantile_genes_df = genes_quantiles(inputdata=norm_indata,lower_quantile=preselect_quantile[0],
                                                upper_quantile=preselect_quantile[1])        
        preselected_genes=preselected_quantile_genes_df.index.tolist() if preselected_genes is None else list(set(preselected_genes).intersection(preselected_quantile_genes_df.index.tolist()))        
    if quantiles:
        norm_indata = genes_quantiles(inputdata=norm_indata, lower_quantile=quantiles[0],upper_quantile=quantiles[1])    
    gene_parameters_df=None if not modification_method=='multiplicative_modification_space' else utils.gene_parameters(norm_indata)    
    purtubed_data_dict=prepare_data(inputdata=norm_indata, fc=fc, no_genes=no_genes,  
                                    no_cells=no_cells, method=modification_method,
                                    quantiles=None,filter_genes_pattern=None,genes_and_cells_cut_off=None,
                                    normalize=None,pseudo_value=1,find_var_genes=None,
                                    preselected_genes=preselected_genes,
                                    gene_modification_profile_df=gene_modification_profile_df,
                                    gene_parameters_df=gene_parameters_df,genes_in_all_cells=genes_in_all_cells)
    mod_genes=purtubed_data_dict['genes']
    mod_cells=purtubed_data_dict['cells']
    mod_data=purtubed_data_dict['data']
    pre_var_genes_mat=filter_genes_dispersion(data=norm_indata.transpose().values,flavor='seurat',min_disp=None,max_disp=None,
                                              min_mean=None, max_mean=None,n_bins=20,n_top_genes=topVarGenes,log=False,
                                              copy=False)
    post_var_genes_mat=filter_genes_dispersion(data=mod_data.transpose().values,flavor='seurat',min_disp=None,max_disp=None,
                                              min_mean=None, max_mean=None,n_bins=20,n_top_genes=topVarGenes,log=False,
                                              copy=False)
    pre_var_genes_df=pd.DataFrame(pre_var_genes_mat,index=norm_indata.index)    
    post_var_genes_df=pd.DataFrame(post_var_genes_mat,index=mod_data.index)    
    top_pre_mod_var_genes_df=pre_var_genes_df[pre_var_genes_df.gene_subset==True]    
    top_mod_var_genes_df=post_var_genes_df[post_var_genes_df.gene_subset==True]    
    mod_and_var_genes=set(mod_genes).intersection(top_mod_var_genes_df.index)    
    pre_var_genes=pre_var_genes_df.index.tolist()    
    pre_mod_colors=['red' if c else 'gray' for c in pre_var_genes_df.gene_subset.tolist()]    
    pre_mod_labs=['variable' if p=='gray' else 'non_var' for p in pre_mod_colors]    
    pre_var_genes_df["col"]=pre_mod_colors    
    pre_var_genes_df["labs"]=pre_mod_labs     
    fig=plt.figure()    
    plt.subplot(2,2,1)    
    plt.scatter(x=np.log2(pre_var_genes_df.dispersions_norm+1),
               y=np.log2(pre_var_genes_df.means+1),
               c=pre_var_genes_df.col,label=pre_var_genes_df.labs,s=10)    
    plt.xlabel('Dispersion')    
    plt.ylabel('Mean expresssion')    
    plt.title('Pre-modification')    
    var_patch = utils.mpatches.Patch(color='red', label='Variable genes')    
    non_var_patch = utils.mpatches.Patch(color='gray', label='Non-variable genes')    
    plt.legend(handles=[var_patch,non_var_patch],fontsize = 7,loc=0)  
    plt.subplot(2,2,2)    
    plt.scatter(x=np.log2(pre_var_genes_df.dispersions_norm.loc[mod_genes]+1),
               y=np.log2(pre_var_genes_df.means.loc[mod_genes]+1),
               c=pre_var_genes_df.col.loc[mod_genes],s=10)    
    plt.xlabel('Dispersion')    
    plt.ylabel('Mean expresssion')    
    plt.title('Pre-modification(target genes)')    
    var_patch = utils.mpatches.Patch(color='red', label='Variable genes')    
    non_var_patch = utils.mpatches.Patch(color='gray', label='Non-variable genes')    
    plt.legend(handles=[var_patch,non_var_patch],fontsize = 7,loc=0)
    post_mod_colors=['black' if c else 'gray' for c in post_var_genes_df.gene_subset.tolist()]    
    post_var_genes=post_var_genes_df.index.tolist()    
    post_mod_final_colors=[]    
    for m in range(len(post_mod_colors)):        
        temp_gene=post_var_genes[m]        
        if temp_gene in mod_and_var_genes and temp_gene in top_pre_mod_var_genes_df.index.tolist():            
            post_mod_final_colors.append('green')        
        elif temp_gene in mod_and_var_genes and temp_gene not in top_pre_mod_var_genes_df.index.tolist():           
            post_mod_final_colors.append('red')          
        elif temp_gene in mod_genes and temp_gene not in  mod_and_var_genes:            
            post_mod_final_colors.append('blue')
        else:
            post_mod_final_colors.append(post_mod_colors[m])
    post_var_genes_df["col"]=post_mod_final_colors
    plt.subplot(2,2,3)
    plt.subplots_adjust(hspace=0.5)
    plt.scatter(x=np.log2(post_var_genes_df.dispersions_norm+1),
               y=np.log2(post_var_genes_df.means+1),
               c=post_mod_final_colors,s=10)
    plt.xlabel('Dispersion')
    plt.ylabel('Mean expresssion')    
    plt.title('Post-modification') 
    var_patch = utils.mpatches.Patch(color='red', label='New variable')
    non_var_patch = utils.mpatches.Patch(color='gray', label='Non-variable genes')
    pre_var_patch = utils.mpatches.Patch(color='black', label='pre-variable')
    post_non_var_patch = utils.mpatches.Patch(color='blue', label='Mod-non varible') 
    pre_post_var_patch = utils.mpatches.Patch(color='green', label='Var both pre- and post- mod.')
    plt.legend(handles=[var_patch,non_var_patch,post_non_var_patch,pre_var_patch,pre_post_var_patch],
             fontsize = 7,loc=0)
    plt.subplot(2,2,4)
    plt.subplots_adjust(hspace=0.5)
    plt.scatter(x=np.log2(post_var_genes_df.dispersions_norm.loc[mod_genes]+1),
               y=np.log2(post_var_genes_df.means.loc[mod_genes]+1),
               c=post_var_genes_df.col.loc[mod_genes],s=10) 
    plt.xlabel('Dispersion')
    plt.ylabel('Mean expresssion')
    plt.title('Post-modification(mod. genes)')
    var_patch = utils.mpatches.Patch(color='red', label='New variable')
    non_var_patch = utils.mpatches.Patch(color='gray', label='Non-variable genes')
    pre_var_patch = utils.mpatches.Patch(color='black', label='pre-variable')
    post_non_var_patch = utils.mpatches.Patch(color='blue', label='Mod-non varible')    
    pre_post_var_patch = utils.mpatches.Patch(color='green', label='Var both pre- and post- mod.')    
    plt.legend(handles=[var_patch,non_var_patch,post_non_var_patch,pre_var_patch,pre_post_var_patch],
             fontsize = 7,loc=0)
    fig.savefig(os.path.join(output_dir,'mean_dispersion_fig.pdf'))
    pass

def plot_genes_dispersion(result,genes=None, log=False, save=None, show=None):
    gene_subset = result.gene_subset
    means = result.means
    dispersions = result.dispersions
    dispersions_norm = result.dispersions_norm
    size = rcParams['figure.figsize']
    plt.figure(figsize=(2*size[0], size[1]))
    plt.subplots_adjust(wspace=0.3)
    for idx, d in enumerate([dispersions_norm, dispersions]):
        plt.subplot(2, 2, idx + 1)
        for label, color, mask in zip(['highly variable genes', 'other genes'],
                                      ['black', 'grey'],
                                      [gene_subset, ~gene_subset]):
            
            if False: means_, disps_ = np.log10(means[mask]), np.log10(d[mask])
            else: means_, disps_ = means[mask], d[mask]
            plt.scatter(means_, disps_, label=label, c=color, s=1)
        if log:  # there's a bug in autoscale
            plt.xscale('log')
            plt.yscale('log')
            min_dispersion = np.min(dispersions)
            y_min = 0.95*min_dispersion if min_dispersion > 0 else 1e-1
            plt.xlim(0.95*np.min(means), 1.05*np.max(means))
            plt.ylim(y_min, 1.05*np.max(dispersions))
        if idx == 0: plt.legend()
        plt.xlabel(('$log_{10}$ ' if False else '') + 'mean expressions of genes')
        plt.ylabel(('$log_{10}$ ' if False else '') + 'dispersions of genes'
                  + (' (normalized)' if idx == 0 else ' (not normalized)'))
    return plt