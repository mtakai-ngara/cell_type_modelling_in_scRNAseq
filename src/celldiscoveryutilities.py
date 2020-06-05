import os,copy,pickle,re,time,itertools,random,numpy as np
import subprocess
import statsmodels.api as sm,math as ma, pandas as pd
from sklearn import  metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import neighbors
from scipy.stats import poisson
from numpy.random import beta, poisson
import multiprocessing as mp
from scipy.stats import binom
from sklearn.manifold import TSNE
#from sklearn.utils import check_array
from sklearn.utils.validation import check_array
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.sparse import issparse
import scanpy.api as scanpi
from scipy.stats import ttest_ind
from scipy import stats

#All utility methods for running the code

#Methods for clustering and reducing dimension in multi-dimensional data

def run_kmeans(input_array,no_clusters=2,scale_array=True):
    ''' Runs K-means clustering on the input array'''
    #np.random.seed(2573780)
    data = scale(input_array) if scale_array else input_array
    kmean = KMeans(init='k-means++', n_clusters=no_clusters, n_init=10).fit(data)
    return kmean

def run_knn(input_array,labels,no_neighbours=5, radius=1.0,algorithm='auto',
            leaf_size=30,distance_metric='minkowski', p=2,cpus=1):
    '''Run the k-nearest neighbours for classification'''
    input_array=np.array(input_array)
    clf = neighbors.KNeighborsClassifier(n_neighbors=no_neighbours,radius=radius,algorithm=algorithm,
                                         leaf_size=leaf_size,metric=distance_metric,p=p,n_jobs=cpus)
    clf.fit(input_array, labels)
    prediction_res = clf.predict(input_array)
    return(prediction_res)

def run_dbscan(input_array,in_min_cluster_size=5,in_min_samples=None, in_metric='euclidean'):
    ''' Runs hdbscan clustering on the input array'''
    #np.random.seed(2573780)
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=in_min_cluster_size,min_samples=in_min_samples,
                                metric=in_metric).fit(input_array)
    return hdbscan_clusterer

def run_pca(input_array,components=None,transpose=False):
    '''Runs PCA on input array'''
    #np.random.seed(2573780)
    in_array= np.array(input_array)
    in_array= in_array if not transpose else np.transpose(in_array)
    in_array_dim=input_array.shape    
    final_component = in_array_dim[0] if components == None or components > in_array_dim[0] else components        
    pca_res_dict={}
    try:
        sklearn_pca = PCA(n_components=final_component)
        sklearn_pca.fit(in_array)
        cumulative_variance = np.cumsum(np.round(sklearn_pca.explained_variance_ratio_, decimals=4) * 100)
        sklearn_tranfrom=sklearn_pca.fit_transform(in_array)        
        pca_res_dict['cum_variance']=cumulative_variance        
        pca_res_dict['pca_array']=sklearn_tranfrom        
        pca_res_dict['explained_variance']=sklearn_pca.explained_variance_ratio_
    except :
        pca_res_dict['pca_array']=np.NAN
        return pca_res_dict
    else:
        return pca_res_dict
    pass

def run_pca_on_df(input_df,components=None,transpose=False):    
    '''Runs PCA reduction on multi-dimensional data frame'''    
    #np.random.seed(2573780)
    input_array= np.array(input_df)  
    final_input_array= input_array if not transpose else np.transpose(input_array)
    final_input_array_dim=final_input_array.shape
    final_component = final_input_array_dim[0] if components == None or components > final_input_array_dim[0] else components    
    try:        
        pca_res_dict=run_pca(input_array=final_input_array,components=final_component)
        pca_array=pca_res_dict['pca_array']
        index=input_df.columns if transpose else input_df.index
        pca_df=pd.DataFrame(pca_array,index=index,columns=['PC'+str(m+1) for m in range(final_component)])
        pca_res_dict['pca_df'] = pca_df        
    except:        
        pca_res_dict={}        
        pca_res_dict['pca_df'] = np.NAN        
    return pca_res_dict

def get_genes_sf(counts_df,pseudo_value=1,loc_func=np.median, return_norm_expr=True):
    '''Gets the size factor and DESeq normalized expression levels from counts data frame'''
    counts_df=counts_df+pseudo_value
    log_genes_mean_series=np.log(counts_df).apply(np.mean,axis=1)
    out_dict={}
    log_counts_df=counts_df.apply(np.log,axis=0)
    log_counts_diff_df=log_counts_df.sub(log_genes_mean_series,axis=0)
    samples_sf=log_counts_diff_df.apply(loc_func,axis=0).apply(np.exp)
    #samples_sf = log_counts_diff_df.median(axis=0,skipna=True).apply(np.exp)
    out_dict['sf'] = samples_sf.to_dict()
    if (return_norm_expr):
        norm_expr_df=counts_df.div(samples_sf,axis=1)
        out_dict['norm'] = norm_expr_df
    return out_dict

def log_transform_dataframe(inputdata,base=2.718281828459045,pseudo_value=1):
    '''Log-transforms data frame using the provided base'''
    logged_array=log_transform_array(input_array=inputdata.values,base=base,pseudo_value=pseudo_value)
    outputdata=pd.DataFrame(logged_array,index=inputdata.index,columns=inputdata.columns)
    return outputdata

def log_transform_array(input_array,base=2.718281828459045,pseudo_value=1):
    '''Log-transforms array using the provided base'''
    logged_array=np.log(input_array+pseudo_value)/np.log(base)
    return logged_array

## Step 1. Functions for reading,transformation and perturbing gene expression dataframe
def read_inputdata(filepath,sep='\t',header=0,index_col=0,nrows=None):
    '''Reads data table into a data frame'''
    input_df=pd.read_csv(filepath,sep=sep,header=header,index_col=index_col,nrows=nrows)
    return (input_df)

def genes_quantiles(inputdata,lower_quantile=0.0,upper_quantile=1.0):
    # returns the input data frame after selecting gene from the specific mean-quantiles 
    mean_series=inputdata.mean(axis=1)
    lower_cut_off=mean_series.quantile(np.float(lower_quantile))
    upper_cut_off = mean_series.quantile(np.float(upper_quantile))
    mean_series_filt=mean_series[mean_series>=lower_cut_off]
    print('Lower cut-off:',lower_cut_off,'Upper cut-off:',upper_cut_off)
    if np.round(upper_quantile,2)<1.00:
        mean_series_filt=mean_series_filt[mean_series_filt < upper_cut_off]
        output_data=inputdata.loc[mean_series_filt.index,inputdata.columns]            
        return  output_data          
    else:        
        mean_series_filt[mean_series_filt <= upper_cut_off]        
        output_data=inputdata.loc[mean_series_filt.index,inputdata.columns]        
        return output_data       
    pass


def plot_expression_data(inputfile,outputdir=None,meta_file=None,meta_col=None,expr_cut_off=1.0,
                         no_cell=1,genes_per_cell_cut_off=1,
                         gene_distribution_quantiles=[[0.00,0.25],[0.25,0.5],[0.50,0.75],[0.75,1.00]],
                         gene_modification_quantiles=[[0.00,0.25],[0.25,0.5],[0.50,0.75],[0.75,1.00]],
                         gene_modification_target_size=10,
                         kde=True,log=True,pseudo_value=1.0,
                         gene_modification_x_axis='quantiles', gene_modification_y_axis='matching_genes_count',
                         gene_modification_in_kind='point', gene_modification_in_col_wrap=3, 
                         gene_modification_hue='fc', gene_modification_col_order='fc',
                         gene_modification_split_groups=None,
                         gene_modification_fc=[2,5,10,20,50,100],gene_modification_mean_interval=1.0,
                         gene_modification_drop_out_interval=0.01,gene_modification_cv2=None,figsize=(12,12)):
    '''Method for plotting an expression matrix. Assumes the data is un-logged'''
    expt_name=inputfile.split('/')[-2]
    if not outputdir:
        outputdir=os.path.join('/'.join(inputfile.split('/')[:-2]),'figures/data_parameter_plots')       
        safe_mkdir(outputdir)        
    else:        
        outputdir=os.path.join(outputdir,'figures/data_parameter_plots')
        safe_mkdir(outputdir)
    norm_expr_df=pd.read_csv(inputfile,sep='\t',header=0,index_col=0)
    selected_gns=norm_expr_df.index[norm_expr_df[norm_expr_df>=expr_cut_off].count(axis=1)>=no_cell]
    selected_samples=norm_expr_df.columns[norm_expr_df[norm_expr_df>=expr_cut_off].count(axis=0)>=genes_per_cell_cut_off]
    norm_expr_df=norm_expr_df.loc[selected_gns,selected_samples]
    mean_expr=np.log(norm_expr_df.mean(axis=1)+pseudo_value)
    drop_rate=1-norm_expr_df[norm_expr_df>0].count(axis=1)/norm_expr_df.shape[1]
    mean_vs_dropout_fig=plot_scatter(x=mean_expr,y=drop_rate,points_size=50,plot_title='Mean vs. dropout',
                                     col_pelette="husl",col_list='blue',xlab='Mean expression',
                                     ylab='Drop out proportion',fig_size=(10,10))
    mean_vs_dropout_fig_fname=os.path.join(outputdir,'mean_vs_dropout.pdf')
    mean_vs_dropout_fig.savefig(mean_vs_dropout_fig_fname)
    log_norm_expr_df=log_transform_dataframe(inputdata=norm_expr_df,pseudo_value=pseudo_value)
    norm_expr_pca_dict=run_pca_on_df(components=10,input_df=log_norm_expr_df,transpose=True)
    pca_df=norm_expr_pca_dict['pca_df']
    if meta_file:
        meta_df=pd.read_csv(meta_file,sep='\t',header=0,index_col=0)        
        temp_samples=set(meta_df.index).intersection(pca_df.index)        
        if len(temp_samples) >=3:        
            meta_df=meta_df.loc[temp_samples,:]
            temp_pca_df=pca_df.loc[temp_samples,:]
            grps=[grp for grp in meta_df.columns if len(meta_df.loc[:,grp].unique())>1 and len(meta_df.loc[:,grp].unique())<=10]
            for grp in grps:
                col_list=list(meta_df.loc[:,grp])
                temp_pca_df[grp]=col_list
                temp_pca_pairs_fig=sns.pairplot(temp_pca_df,plot_kws={'s':80},hue=grp)
                temp_pca_fig_fname=os.path.join(outputdir,grp+'_pca.pdf')
                temp_pca_pairs_fig.savefig(temp_pca_fig_fname)              
    pca_pairs_fig=sns.pairplot(pca_df,plot_kws={'s':80})    
    pca_fig_fname=os.path.join(outputdir,'pca.pdf')
    pca_pairs_fig.savefig(pca_fig_fname)
    genes_mean=norm_expr_df.mean(axis=1)  
    genes_dist_fig=plt.figure(figsize=figsize)   
    quant_dict={}    
    for n in range(len(gene_distribution_quantiles)):
        quant=gene_distribution_quantiles[n]
        
        temp_norm_expr_df=genes_quantiles(inputdata=norm_expr_df,lower_quantile=quant[0],
                                          upper_quantile=quant[1])        
        quant_dict['-'.join([str(t) for t in quant])]=temp_norm_expr_df.index        
        expr=temp_norm_expr_df.melt().value        
        subplot_tracker=n+1        
        plt.subplot(2,2,subplot_tracker)        
        if log==True:            
            expr=np.log2(expr+pseudo_value)        
        temp_hist=sns.distplot(expr,kde=kde)        
        plt.xlabel('Expression(log)' if log==True else 'Expression')        
        plt.ylabel('Frequency' if kde!=True else 'Density')        
        plt.title('Quantile: '+'-'.join([str(q) for q in quant]))      
    temp_his_fname=os.path.join(outputdir,'genes_expression_distribution.pdf')  
    genes_dist_fig.savefig(temp_his_fname)    
    indata_params_df=gene_parameters(norm_expr_df)    
    gene_mod_results_list=[]    
    for gmq in gene_modification_quantiles:        
        temp_norm_expr_df=genes_quantiles(inputdata=norm_expr_df,lower_quantile=gmq[0],
                                          upper_quantile=gmq[1])
        print('Gene modification quantile:',gmq)
        print('Expression shape:',temp_norm_expr_df.shape)     
        target_genes=np.random.choice(indata_params_df.index,gene_modification_target_size,replace=False)        
        pool = mp.Pool(processes=10)      
        temp_args=[(indata_params_df,gn,f,None) for f in gene_modification_fc for gn in target_genes]        
        temp_results = pool.starmap_async(genes_with_same_parameters, iterable=temp_args).get()        
        pool.close()        
        pool.join()        
        del temp_args      
        for temp_result_dict in temp_results:
            gene_mod_results_list.append(['-'.join([str(q) for q in gmq]),temp_result_dict['gene'],
                                          temp_result_dict['fc'],len(temp_result_dict['matching_genes_list']),
                                          temp_result_dict['matching_genes_list'],
                                          temp_result_dict['mean_interval'],temp_result_dict['mean'],
                                          temp_result_dict['dropout'],temp_norm_expr_df.shape[0]])
            '''
            gene_mod_results_list.append(['-'.join([str(q) for q in gmq]),temp_result_dict['gene'],
                                          
                                          temp_result_dict['fc'],temp_result_dict['df'].shape[0],
                                          list(temp_result_dict['df'].index),
                                          temp_result_dict['mean_interval'],temp_result_dict['drop_out_interval'],
                                          temp_result_dict['cv'],temp_result_dict['mean'],
                                          temp_result_dict['dropout'],temp_norm_expr_df.shape[0]])
                                          '''            
    gene_mod_results_df=pd.DataFrame(gene_mod_results_list,
                                     columns=['quantiles','gene','fc','matching_genes_count','matching_genes',
                                              'mean_interval','mean','dropout','quantile_size'])   
    gene_mod_subspace_fname=os.path.join('/'.join(outputdir.split('/')[:-2]),'quantile_gene_mod_subspace.tab')    
    gene_mod_results_df.to_csv(gene_mod_subspace_fname,header=True,index=True,sep='\t')  
    fig_fname=plot_genes_modification_subspace_distribution(subspace_infile=gene_mod_subspace_fname,
                                                            output_dir=outputdir, x_axis=gene_modification_x_axis,
                                                            y_axis=gene_modification_y_axis,
                                                            in_kind=gene_modification_in_kind, 
                                                            in_col_wrap=gene_modification_in_col_wrap, 
                                                            hue=gene_modification_hue, col_order=gene_modification_col_order,
                                                            split_groups=gene_modification_split_groups)
    pass

def plot_scatter(x,y,points_size=50,plot_title='Scatter plot',col_pelette="husl",col_list=None,
                 xlab='X-axis',ylab='Y-axis',fig_size=(10,10)):
    '''Plots a scatter '''
    input_col_list=col_list if col_list!=None else [1 for m in range(len(x))]
    fig=plt.figure(figsize=fig_size)
    plt.scatter(x=x, y=y,c=input_col_list,s=points_size)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(plot_title)
    return fig

def plot_homogeneity_score_distribution(homo_scores_infile, output_dir, x_axis='genes_count', y_axis='hs',
                                        in_kind='point', in_col_wrap=3, hue='cells_count', col_order='fc',
                                        split_groups=None,replicates_per_grp=10,
                                        replicates_grps=['method', 'cells_count', 'genes_count', 'fc', 'modification_method',
                                              'clustering_method', 'randomize_cells_labels']):
    '''Plots the HS score distributions generated from the run_modification_methods_and_score_in_parallel'''
    safe_mkdir(output_dir)
    expt_name = homo_scores_infile.split('/')[-2]
    homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    figs_dict = {}
    if split_groups:
        split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
        grouped_df = homo_scores_df.groupby(split_groups)
        grps_dict = grouped_df.groups
        for key in grps_dict.keys():            
            try:
                temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]
                sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]
                temp_facegrid = sns.factorplot(x=x_axis, y=y_axis, hue=hue, col=col_order,
                                              col_wrap=in_col_wrap, col_order=sorted_order,
                                              data=temp_homo_scores_df, kind=in_kind)
                temp_facegrid.set_xticklabels(rotation=70)
                group = '_'.join([str(k) for k in key])
                temp_out_fname = os.path.join(output_dir, expt_name+'_'+group +'_hs_score.pdf')
                if os.path.exists(temp_out_fname):
                    subprocess.check_output(['rm' ,temp_out_fname],shell=False)
                temp_facegrid.savefig(temp_out_fname)
                figs_dict[group+'hs_score'] = temp_out_fname                
            except ValueError:               
                pass
    else:
        sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
        temp_facegrid = sns.factorplot(x=x_axis, y=y_axis, hue=hue, col=col_order,
                                      col_wrap=in_col_wrap, col_order=sorted_order,
                                      data=homo_scores_df, kind=in_kind, size=4, aspect=1)
        temp_facegrid.set_xticklabels(rotation=70)
        temp_out_fname = os.path.join(output_dir, expt_name+'_hs_score.pdf')        
        if os.path.exists(temp_out_fname):            
            subprocess.check_output(['rm' ,temp_out_fname],shell=False)
        temp_facegrid.savefig(temp_out_fname)
        figs_dict['hs_score'] = temp_out_fname
    return figs_dict

def plot_var_and_modified_genes_intersection(homo_scores_infile, output_dir=None, x_axis='genes_count', 
                                             y_axis='len_mod_and_var_genes',in_kind='point', in_col_wrap=3,
                                             hue='cells_count',hue_palette=None, col_order='fc',
                                             split_groups=None, replicates_per_grp=10,col_subset=None,
                                             replicates_grps=['method', 'cells_count', 'genes_count', 'fc',
                                                              'modification_method','clustering_method',
                                                              'randomize_cells_labels'],
                                             show_reps_counts=None,facet_size=4):
    '''Plots the intersection between modified and variable genes from the modification based on variable genes'''
    if output_dir:
        safe_mkdir(output_dir)
    expt_name = None
    homo_scores_df=None
    try:
        if os.path.exists(homo_scores_infile):
            expt_name = homo_scores_infile.split('/')[-2]
            homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)
            
    except TypeError:
        homo_scores_df=pd.DataFrame(homo_scores_infile)
    if col_subset:
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]
    homo_scores_df = filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                               replicates_cut_off=replicates_per_grp,
                                               grps=replicates_grps)    
    if homo_scores_df is None: 
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        return     
    else:
        figs_dict = {}
        if split_groups:
            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
            grouped_df = homo_scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for key in grps_dict.keys():
                temp_homo_scores_df = homo_scores_df.iloc[grps_dict[key], :]
                sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]
                if show_reps_counts:
                    temp_homo_scores_dict=temp_homo_scores_df.groupby(show_reps_counts).groups
                    for t_k in temp_homo_scores_dict.keys():
                        print('Reps per grp plotted:',len(temp_homo_scores_dict[t_k]))
                temp_facegrid = sns.catplot(x=x_axis, y=y_axis, hue=hue, col=col_order,col_wrap=in_col_wrap,
                                            col_order=sorted_order,data=temp_homo_scores_df, kind=in_kind,
                                            height=facet_size,palette=hue_palette)
                fig_title='_'.join([str(ky) for ky in key])
                print(fig_title)
                temp_facegrid.set_xticklabels(rotation=70)
                if output_dir:
                    group = '_'.join([str(k) for k in key])
                    temp_out_fname = os.path.join(output_dir,  group + '_mod_vs_var_genes.pdf')
                    if os.path.exists(temp_out_fname):
                        subprocess.check_output(['rm', temp_out_fname], shell=False)
                        
                    temp_facegrid.savefig(temp_out_fname)
                    figs_dict[group + 'hs_score'] = temp_out_fname                    
                else:
                    print('Fig not saved')
        else:
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
            temp_facegrid = sns.factorplot(x=x_axis, y=y_axis, hue=hue, col=col_order,
                                          col_wrap=in_col_wrap, col_order=sorted_order,
                                          data=homo_scores_df, kind=in_kind, size=4, aspect=1)
            temp_facegrid.set_xticklabels(rotation=70)
            temp_out_fname = os.path.join(output_dir, 'mod_vs_var_genes.pdf')
            if os.path.exists(temp_out_fname):
                subprocess.check_output(['rm', temp_out_fname], shell=False)
            temp_facegrid.savefig(temp_out_fname)
            figs_dict['hs_score'] = temp_out_fname
        return figs_dict    
    pass

def plot_mean_homogeneity_score_distribution(homo_scores_infile, output_dir=None, x_axis_lab='fc', y_axis_lab='genes_count',
                                             col_order='fc',split_groups=None,replicates_per_grp=10,
                                             replicates_grps=['method', 'cells_count',  'modification_method',
                                                              'clustering_method', 'randomize_cells_labels'],
                                             xlabs_ratotation=70, ylabs_ratotation=0,col_subset=None,in_cmap='seismic',
                                            add_column_dict=None,show_reps_counts=None):    
    '''Plots the HS score distributions generated from the run_modification_methods_and_score_in_parallel'''    
    homo_scores_df=None    
    expt_name=None
    try:
        if os.path.exists(homo_scores_infile):
            expt_name = homo_scores_infile.split('/')[-2]
            homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)            
    except TypeError:
        homo_scores_df=pd.DataFrame(homo_scores_infile)    
    if col_subset:        
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]  
    homo_scores_df=homo_scores_df.astype({'hs':str})    
    homo_scores_df=homo_scores_df[homo_scores_df.hs!='nan']    
    homo_scores_df=homo_scores_df.astype({'hs':np.float})   
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    if add_column_dict:
        for k,v in add_column_dict.items():
            homo_scores_df[k]=v            
    if homo_scores_df is None:         
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        return 
    else:        
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)        
        fc_list=homo_scores_df.fc.unique().tolist()        
        if 0.0 in fc_list:            
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]            
            temp_fc_list.append(0.0)            
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])            
            fc_list=temp_fc_list
        else:
            fc_list=sorted(fc_list)
        last_fig_fname=None
        if split_groups:
            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
            grouped_df = homo_scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for key in grps_dict.keys():
                temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]
                if show_reps_counts:
                    temp_homo_scores_dict=temp_homo_scores_df.groupby(show_reps_counts).groups
                    for t_k in temp_homo_scores_dict.keys():
                        print('Reps per grp plotted:',len(temp_homo_scores_dict[t_k]))                    
                mean_score=[]
                for t_fc in fc_list:
                    temp_list=[]
                    for gc in gene_cts:                        
                        mean_sc_score=temp_homo_scores_df[(temp_homo_scores_df.fc==t_fc) &
                                                  (temp_homo_scores_df.genes_count==gc)].loc[:,'hs'].mean(axis=0)                
                        temp_list.append(mean_sc_score)
                    mean_score.append(temp_list)
                mean_hs_fig=plt.figure(figsize=(10,10))
                group = '_'.join(sorted([str(k) for k in key]))                
                temp_mean_score_df=pd.DataFrame(mean_score,columns=gene_cts,index=fc_list)                
                heat_ax=plot_heatmap(data_frame=temp_mean_score_df.transpose(),xlab=x_axis_lab,
                             ylab=y_axis_lab,labs_font_size=10,xlabs_ratotation=xlabs_ratotation,
                             ylabs_ratotation=ylabs_ratotation,cmap=in_cmap)
                plt.title(group)                
                if output_dir:
                    safe_mkdir(output_dir)
                    fig_fname=os.path.join(output_dir,group+'_mean_hs_score.pdf')
                    mean_hs_fig.savefig(fig_fname)
                    last_fig_fname=fig_fname
        else:
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
            print('To-do:')
        return last_fig_fname
    pass

def plot_mean_cluster_score_heatmap(homo_scores_infile,score='hs', output_dir=None,
                                    x_axis_lab='fc', y_axis_lab='genes_count',col_order='fc',split_groups=None,
                                    replicates_per_grp=10,replicates_grps=['method', 'cells_count',  
                                                                           'modification_method','clustering_method', 
                                                                           'randomize_cells_labels'],
                                    xlabs_ratotation=70, ylabs_ratotation=0,col_subset=None,in_cmap='seismic',
                                    add_column_dict=None,show_reps_counts=None,limit=None,fig_title=None):    
    '''Plots the score distributions generated from the run_modification_methods_and_score_in_parallel'''
    homo_scores_df=None
    expt_name=None
    try:
        if os.path.exists(homo_scores_infile):
            expt_name = homo_scores_infile.split('/')[-2]
            homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)            
    except TypeError:
        homo_scores_df=pd.DataFrame(homo_scores_infile)
    if col_subset:        
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]
    homo_scores_df=homo_scores_df.astype({score:str})
    homo_scores_df=homo_scores_df[homo_scores_df[score]!='nan']
    homo_scores_df=homo_scores_df.astype({score:np.float})
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    if add_column_dict:
        for k,v in add_column_dict.items():
            homo_scores_df[k]=v            
    if homo_scores_df is None: 
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        return 
    else:
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)
        fc_list=homo_scores_df.fc.unique().tolist()
        if 0.0 in fc_list:
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]
            temp_fc_list.append(0.0)
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])
            fc_list=temp_fc_list
        else:
            fc_list=sorted(fc_list)
        last_fig_fname=None
        if split_groups:
            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
            grouped_df = homo_scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for key in grps_dict.keys():
                temp_homo_scores_df = homo_scores_df.iloc[grps_dict[key], :].reset_index(drop=True)
                mean_score=[]
                for t_fc in fc_list[:]:
                    temp_list=[]
                    for gc in gene_cts[:]: 
                        temp_fc_gn_homo_scores_df=temp_homo_scores_df[(temp_homo_scores_df.fc==t_fc) & (temp_homo_scores_df.genes_count==gc)]
                        tmp_rand_index=None
                        if limit:
                            tmp_rand_index=np.random.choice(temp_fc_gn_homo_scores_df.index, size=limit,replace=False)
                            temp_fc_gn_homo_scores_df=temp_fc_gn_homo_scores_df.loc[tmp_rand_index,:]
                            
                        if show_reps_counts:
                            print(temp_fc_gn_homo_scores_df.shape[0])
                        mean_sc_score=temp_fc_gn_homo_scores_df.loc[:,score].mean(axis=0)
                        #print(len(temp_fc_gn_homo_scores_df.loc[:,score]))
                        temp_list.append(mean_sc_score)
                    mean_score.append(temp_list)
                mean_hs_fig=plt.figure(figsize=(10,10))
                group = '_'.join(sorted([str(k) for k in key]))                
                temp_mean_score_df=pd.DataFrame(mean_score,columns=gene_cts,index=fc_list)
                heat_ax=plot_heatmap(data_frame=temp_mean_score_df.transpose(),xlab=x_axis_lab,
                             ylab=y_axis_lab,labs_font_size=10,xlabs_ratotation=xlabs_ratotation,
                             ylabs_ratotation=ylabs_ratotation,cmap=in_cmap)
                #group=group+'_'+in_cmap
                if fig_title is None:
                    plt.title(group) 
                else:
                    plt.title(fig_title)                     
                if output_dir:
                    output_dir=output_dir+'/'+score
                    safe_mkdir(output_dir)
                    if fig_title is None:
                        fig_fname=os.path.join(output_dir,group+'_mean_'+score+'_score.pdf')  
                    else:
                        fig_fname=os.path.join(output_dir,re.sub(pattern='[\s\t]+',repl='_',string=fig_title)+'.pdf')    
                    mean_hs_fig.savefig(fig_fname)
                    last_fig_fname=fig_fname
        else:
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
            print('To-do:')
        return last_fig_fname
    pass

def plot_tpr_vs_fpr(homo_scores_infile, output_dir, x_axis_lab='fpr', y_axis_lab='tpr',col_order='fc',
                            split_groups=None,replicates_per_grp=10,
                            replicates_grps=['method', 'cells_count',  'modification_method',
                                             'clustering_method', 'randomize_cells_labels'],
                            hue='genes_count',xlabs_ratotation=70, ylabs_ratotation=0,in_col_wrap=4,
                            col_subset=None,hue_subset=None,kind='scatter'):    
    '''Plots the tpr vs. fpr per replicate group'''
    safe_mkdir(output_dir)
    expt_name = homo_scores_infile.split('/')[-2]
    homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)    
    homo_scores_df=homo_scores_df.astype({'hs':str})    
    homo_scores_df=homo_scores_df[homo_scores_df.hs!='nan']    
    homo_scores_df=homo_scores_df.astype({'hs':np.float})   
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    if homo_scores_df is None:         
        print('No replicates passed the required cut-off of: ',replicates_per_grp)        
        return 
    else:
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)        
        fc_list=homo_scores_df.fc.unique().tolist()        
        if 0.0 in fc_list:           
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]            
            temp_fc_list.append(0.0)            
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])            
            fc_list=temp_fc_list           
        else:            
            fc_list=sorted(fc_list)
        if col_subset:            
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]            
        if hue_subset:            
            homo_scores_df=homo_scores_df[homo_scores_df[hue].isin(hue_subset)]            
        homo_scores_df=pd.DataFrame(homo_scores_df.values,columns=homo_scores_df.columns,
                       index=[ind for ind in range(homo_scores_df.shape[0])])        
        last_fig_fname=None
        if split_groups:
            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
            grouped_df = homo_scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for key in grps_dict.keys():                
                if kind=='scatter':                    
                    temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]
                    sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]                
                    group='_'.join(sorted([str(k) for k in key]))                
                    g = sns.FacetGrid(temp_homo_scores_df, col=col_order,col_wrap=in_col_wrap,hue=hue)                
                    g = (g.map(plt.scatter, x_axis_lab, y_axis_lab, edgecolor='none').add_legend())    
                    temp_out_fname = os.path.join(output_dir, group + '_tpr_vs_fpr.pdf')                
                    g.savefig(temp_out_fname)
                    plt.title(group)                
                    last_fig_fname=temp_out_fname                    
                elif kind=='summary':                    
                    temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]                    
                    group='_'.join(sorted([str(k) for k in key]))
                    sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]                 
                    unique_hue=temp_homo_scores_df[hue].unique()                    
                    unique_col_order=temp_homo_scores_df[col_order].unique()                    
                    col_list=[]                    
                    hue_list=[]                    
                    tpr_list=[]                   
                    fpr_list=[]                    
                    for m in unique_col_order:                        
                        for n in unique_hue:                            
                            subset_temp_homo_scores_df=temp_homo_scores_df[(temp_homo_scores_df[hue]==n)&(temp_homo_scores_df[col_order]==m)]                            
                            mean_tpr=np.round(subset_temp_homo_scores_df[y_axis_lab].mean(),2)                            
                            mean_fpr=np.round(subset_temp_homo_scores_df[x_axis_lab].mean(),2)                            
                            tpr_list.append(mean_tpr)                            
                            fpr_list.append(mean_fpr)                            
                            hue_list.append(n)                            
                            col_list.append(m)                           
                    temp_df=pd.DataFrame(np.array([col_list, hue_list,tpr_list,fpr_list]).transpose(),
                                       columns=[col_order,hue,y_axis_lab,x_axis_lab])                  
                    temp_fig = sns.factorplot(x=x_axis_lab, y=y_axis_lab,data=temp_df,
                                             col=col_order,col_wrap=in_col_wrap)                    
                    for ax in temp_fig.axes.flat:                        
                        ax.plot((0, 1), (0, 1), c=".2", ls="--")                    
                    temp_fig.set_xticklabels(rotation=70)                
                    temp_out_fname = os.path.join(output_dir, group + '_tpr_vs_fpr.pdf')                
                    temp_fig.savefig(temp_out_fname)
                    last_fig_fname=temp_out_fname                 
                else:                    
                    temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]
                    sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]                 
                    temp_homo_scores_df=temp_homo_scores_df.sort_values(by=[hue],ascending=True)                
                    group='_'.join(sorted([str(k) for k in key]))                
                    #g = sns.FacetGrid(temp_homo_scores_df, col='fc',col_wrap=3,hue=hue)                
                    #g = (g.map(plt.scatter, x_axis_lab, y_axis_lab, edgecolor='none').add_legend())    
                    temp_out_fname = os.path.join(output_dir, group + '_tpr_vs_fpr.pdf')                
                    g.savefig(temp_out_fname)
                    plt.title(group)                
                    last_fig_fname=temp_out_fname                
        else:            
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]            
            print('To do....')
        return last_fig_fname    
    pass

def plot_tpr_vs_fdr(homo_scores_infile, output_dir, x_axis_lab='fdr', y_axis_lab='tpr',col_order='fc',
                            split_groups=None,replicates_per_grp=10,
                            replicates_grps=['method', 'cells_count',  'modification_method',
                                             'clustering_method', 'randomize_cells_labels'],
                            hue='genes_count',xlabs_ratotation=70, ylabs_ratotation=0,in_col_wrap=4,
                            col_subset=None,hue_subset=None,kind='scatter'):    
    '''Plots the tpr vs. fpr per replicate group'''
    safe_mkdir(output_dir)
    expt_name = homo_scores_infile.split('/')[-2]
    homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)    
    homo_scores_df=homo_scores_df.astype({'hs':str})    
    homo_scores_df=homo_scores_df[homo_scores_df.hs!='nan']    
    homo_scores_df=homo_scores_df.astype({'hs':np.float})   
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    if homo_scores_df is None:         
        print('No replicates passed the required cut-off of: ',replicates_per_grp)        
        return   
    else: 
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)        
        fc_list=homo_scores_df.fc.unique().tolist()        
        if 0.0 in fc_list:            
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]           
            temp_fc_list.append(0.0)            
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])           
            fc_list=temp_fc_list           
        else:            
            fc_list=sorted(fc_list)
        if col_subset:            
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]            
        if hue_subset:            
            homo_scores_df=homo_scores_df[homo_scores_df[hue].isin(hue_subset)]        
        last_fig_fname=None
        if split_groups:
            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
            grouped_df = homo_scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for key in grps_dict.keys():                
                if kind=='scatter':                    
                    temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]
                    sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]                
                    group='_'.join(sorted([str(k) for k in key]))                
                    g = sns.FacetGrid(temp_homo_scores_df, col=col_order,col_wrap=in_col_wrap,hue=hue)                
                    g = (g.map(plt.scatter, x_axis_lab, y_axis_lab, edgecolor='none').add_legend())    
                    temp_out_fname = os.path.join(output_dir, group + '_tpr_vs_fdr.pdf')                
                    g.savefig(temp_out_fname)
                    plt.title(group)                
                    last_fig_fname=temp_out_fname                    
                elif kind=='summary':                    
                    temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]                    
                    group='_'.join(sorted([str(k) for k in key]))
                    sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]                 
                    unique_hue=temp_homo_scores_df[hue].unique()                    
                    unique_col_order=temp_homo_scores_df[col_order].unique()                    
                    col_list=[]                    
                    hue_list=[]                    
                    tpr_list=[]                    
                    fpr_list=[]                    
                    for m in unique_col_order:                        
                        for n in unique_hue:                            
                            subset_temp_homo_scores_df=temp_homo_scores_df[(temp_homo_scores_df[hue]==n)&(temp_homo_scores_df[col_order]==m)]
                            
                            mean_tpr=np.round(subset_temp_homo_scores_df[y_axis_lab].mean(),2)
                            
                            mean_fpr=np.round(subset_temp_homo_scores_df[x_axis_lab].mean(),2)
                            
                            tpr_list.append(mean_tpr)
                            
                            fpr_list.append(mean_fpr)
                            
                            hue_list.append(n)
                            
                            col_list.append(m)
                           
                    temp_df=pd.DataFrame(np.array([col_list, hue_list,tpr_list,fpr_list]).transpose(),
                                       columns=[col_order,hue,y_axis_lab,x_axis_lab])
                  
                    temp_fig = sns.factorplot(x=x_axis_lab, y=y_axis_lab,data=temp_df,
                                             col=col_order,col_wrap=in_col_wrap)
                    
                    temp_fig.set_xticklabels(rotation=70)
                
                    temp_out_fname = os.path.join(output_dir, group + '_tpr_vs_fdr.pdf')
                
                    temp_fig.savefig(temp_out_fname)

                    last_fig_fname=temp_out_fname
                 
                else:
                    
                    temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]

                    sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())] 
                
                    temp_homo_scores_df=temp_homo_scores_df.sort_values(by=[hue],ascending=True)
                
                    group='_'.join(sorted([str(k) for k in key]))
                
                    #g = sns.FacetGrid(temp_homo_scores_df, col='fc',col_wrap=3,hue=hue)
                
                    #g = (g.map(plt.scatter, x_axis_lab, y_axis_lab, edgecolor='none').add_legend())
    
                    temp_out_fname = os.path.join(output_dir, group + '_tpr_vs_fpr.pdf')
                
                    g.savefig(temp_out_fname)

                    plt.title(group)
                
                    last_fig_fname=temp_out_fname
                    
                
        else:
            
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
            
            print('To do....')

        return last_fig_fname
    
    pass


def plot_mean_accuracy_heatmap(homo_scores_infile, output_dir, x_axis_lab='fc', y_axis_lab='genes_count',
                                             col_order='fc',split_groups=None,replicates_per_grp=10,
                                             replicates_grps=['method', 'cells_count',  'modification_method',
                                                              'clustering_method', 'randomize_cells_labels'],
                                             xlabs_ratotation=70, ylabs_ratotation=0,col_subset=None,in_cmap='seismic'):
    
    '''Plots the HS score distributions generated from the 
    run_modification_methods_and_score_in_parallel'''

    safe_mkdir(output_dir)

    expt_name = homo_scores_infile.split('/')[-2]

    homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)
    
    if col_subset:
        
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]
       
    
    homo_scores_df=homo_scores_df.astype({'acc':str})
    
    homo_scores_df=homo_scores_df[homo_scores_df.acc!='nan']
    
    homo_scores_df=homo_scores_df.astype({'acc':np.float})
   
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    if homo_scores_df is None: 
        
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        
        return 
    
    
    else:
        
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)

        fc_list=homo_scores_df.fc.unique().tolist()
        
        if 0.0 in fc_list:
            
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]
            
            temp_fc_list.append(0.0)
            
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])
            
            fc_list=temp_fc_list
           
        else:
            
            fc_list=sorted(fc_list)
        
        
        last_fig_fname=None

        if split_groups:

            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))

            grouped_df = homo_scores_df.groupby(split_groups)

            grps_dict = grouped_df.groups

            for key in grps_dict.keys():

                temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]

                mean_score=[]

                for t_fc in fc_list:

                    temp_list=[]

                    for gc in gene_cts:
                        
                        mean_sc_score=temp_homo_scores_df[(temp_homo_scores_df.fc==t_fc) &
                                                  (temp_homo_scores_df.genes_count==gc)].loc[:,'acc'].mean(axis=0)
                        
                        temp_list.append(mean_sc_score)

                    mean_score.append(temp_list)

                mean_hs_fig=plt.figure(figsize=(10,10))

                group = '_'.join(sorted([str(k) for k in key]))
                
                temp_mean_score_df=pd.DataFrame(mean_score,columns=gene_cts,index=fc_list)

                heat_ax=plot_heatmap(data_frame=temp_mean_score_df.transpose(),xlab=x_axis_lab,
                             ylab=y_axis_lab,labs_font_size=10,xlabs_ratotation=xlabs_ratotation,
                             ylabs_ratotation=ylabs_ratotation,cmap=in_cmap)

                plt.title(group)

                fig_fname=os.path.join(output_dir,group+'_mean_accuracy_score.pdf')

                mean_hs_fig.savefig(fig_fname)

                last_fig_fname=fig_fname


        else:

            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]


        return last_fig_fname
    
    pass



def plot_mean_fdr_heatmap(homo_scores_infile, output_dir, x_axis_lab='fc', y_axis_lab='genes_count',
                                             col_order='fc',split_groups=None,replicates_per_grp=10,
                                             replicates_grps=['method', 'cells_count',  'modification_method',
                                                              'clustering_method', 'randomize_cells_labels'],
                                             xlabs_ratotation=70, ylabs_ratotation=0,col_subset=None,in_cmap='seismic'):
    
    '''Plots the HS score distributions generated from the 
    run_modification_methods_and_score_in_parallel'''

    safe_mkdir(output_dir)

    expt_name = homo_scores_infile.split('/')[-2]

    homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)
    
    if col_subset:
        
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]
       
    
    homo_scores_df=homo_scores_df.astype({'fdr':str})
    
    homo_scores_df=homo_scores_df[homo_scores_df.fdr!='nan']
    
    homo_scores_df=homo_scores_df.astype({'fdr':np.float})
   
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    if homo_scores_df is None: 
        
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        
        return 
    
    
    else:
        
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)

        fc_list=homo_scores_df.fc.unique().tolist()
        
        if 0.0 in fc_list:
            
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]
            
            temp_fc_list.append(0.0)
            
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])
            
            fc_list=temp_fc_list
           
        else:
            
            fc_list=sorted(fc_list)
        
        
        last_fig_fname=None

        if split_groups:

            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))

            grouped_df = homo_scores_df.groupby(split_groups)

            grps_dict = grouped_df.groups

            for key in grps_dict.keys():

                temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]

                mean_score=[]

                for t_fc in fc_list:

                    temp_list=[]

                    for gc in gene_cts:
                        
                        mean_sc_score=temp_homo_scores_df[(temp_homo_scores_df.fc==t_fc) &
                                                  (temp_homo_scores_df.genes_count==gc)].loc[:,'fdr'].mean(axis=0)
                        
                        temp_list.append(mean_sc_score)

                    mean_score.append(temp_list)

                mean_hs_fig=plt.figure(figsize=(10,10))

                group = '_'.join(sorted([str(k) for k in key]))
                
                temp_mean_score_df=pd.DataFrame(mean_score,columns=gene_cts,index=fc_list)
                heat_ax=plot_heatmap(data_frame=temp_mean_score_df.transpose(),xlab=x_axis_lab,
                             ylab=y_axis_lab,labs_font_size=10,xlabs_ratotation=xlabs_ratotation,
                             ylabs_ratotation=ylabs_ratotation,cmap=in_cmap)
                plt.title(group)
                fig_fname=os.path.join(output_dir,group+'_mean_fdr_score.pdf')
                mean_hs_fig.savefig(fig_fname)
                last_fig_fname=fig_fname
        else:
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
        return last_fig_fname
    
    pass

def plot_mean_score_heatmap(homo_scores_infile, output_dir,score='hs', x_axis_lab='fc', y_axis_lab='genes_count',
                            col_order='fc',split_groups=None,replicates_per_grp=10,
                            replicates_grps=['method', 'cells_count',  'modification_method',
                                             'clustering_method', 'randomize_cells_labels'],
                            xlabs_ratotation=70, ylabs_ratotation=0,col_subset=None,in_cmap='seismic'):
    '''Plots the HS score distributions generated from the run_modification_methods_and_score_in_parallel'''
    safe_mkdir(output_dir)
    expt_name = homo_scores_infile.split('/')[-2]
    homo_scores_df = pd.read_csv(homo_scores_infile, sep='\t', header=0)
    if col_subset:
            homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]
    homo_scores_df=homo_scores_df.astype({score:str})
    homo_scores_df=homo_scores_df[homo_scores_df[score]!='nan']
    homo_scores_df=homo_scores_df.astype({score:np.float})
    homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                              replicates_cut_off=replicates_per_grp,
                                              grps=replicates_grps)
    
    if homo_scores_df is None: 
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        return 
    else:
        gene_cts=sorted(homo_scores_df.genes_count.unique().tolist(),reverse=True)
        fc_list=homo_scores_df.fc.unique().tolist()
        if 0.0 in fc_list:
            temp_fc_list=[fl for fl in fc_list if fl<1 and fl >0]
            temp_fc_list.append(0.0)
            temp_fc_list.extend([fl for fl in fc_list if fl>=1])
            fc_list=temp_fc_list
        else:
            fc_list=sorted(fc_list)
        last_fig_fname=None
        if split_groups:
            split_groups = list(set(split_groups).intersection(homo_scores_df.columns))
            grouped_df = homo_scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for key in grps_dict.keys():
                temp_homo_scores_df = homo_scores_df.loc[grps_dict[key], :]
                mean_score=[]
                for t_fc in fc_list:
                    temp_list=[]
                    for gc in gene_cts:
                        mean_sc_score=temp_homo_scores_df[(temp_homo_scores_df.fc==t_fc) &
                                                  (temp_homo_scores_df.genes_count==gc)].loc[:,score].mean(axis=0)                       
                        temp_list.append(mean_sc_score)
                    mean_score.append(temp_list)
                mean_hs_fig=plt.figure(figsize=(10,10))
                group = '_'.join(sorted([str(k) for k in key]))                
                temp_mean_score_df=pd.DataFrame(mean_score,columns=gene_cts,index=fc_list)
                heat_ax=plot_heatmap(data_frame=temp_mean_score_df.transpose(),xlab=x_axis_lab,
                             ylab=y_axis_lab,labs_font_size=10,xlabs_ratotation=xlabs_ratotation,
                             ylabs_ratotation=ylabs_ratotation,cmap=in_cmap)
                plt.title(group)
                fig_fname=os.path.join(output_dir,group+'_'+score+'_mean_score.pdf')
                mean_hs_fig.savefig(fig_fname)
                last_fig_fname=fig_fname
        else:
            sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
        return last_fig_fname
    pass

def plot_classification_scores_between_experiments(expriment_path,out_dir=None,x_axis='genes_count',y_axis='hs',
                                                   replicates_per_grp=40,split_groups=None,col_order='fc',
                                                   x_axis_subset=None,hue='top_var_genes_included',
                                                   replicates_grps=None,in_kind='box',
                                                   in_col_wrap=4,ci=95,col_subset=None,hue_order=None,col_order_value=None,
                                                   add_column_dict=None,show_reps_counts=None,palette=None,
                                                   facet_size=4,add_swarm=False,swarm_points_size=4,x_axis_rotation=0,
                                                   subplot_adjust=.95,subplot_title_size=14,no_boots=1000,
                                                   sharex=True,sharey=True,boxplot_mean=True,boxplot_outliers=True,
                                                   boxprops=None, limit=None,errwidth=None,capsize=None, fig_title=None):
    '''Plots the classification scores between experiments'''
    homo_scores_df=None
    try:
        if os.path.exists(expriment_path):
            hs_score_fnames=get_list_of_specific_file_type(directory=expriment_path,file_pattern='hs_scores.tab')
            df_list=[]
            for hs_score_fname in hs_score_fnames:
                temp_hs_score_df=read_inputdata(hs_score_fname)
                df_list.append(temp_hs_score_df)
            homo_scores_df=pd.concat(df_list,ignore_index=True)
    except TypeError:
        homo_scores_df=pd.DataFrame(expriment_path)        
    if col_subset:
        homo_scores_df=homo_scores_df[homo_scores_df[col_order].isin(col_subset)]       
    if x_axis_subset:        
        homo_scores_df=homo_scores_df[homo_scores_df[x_axis].isin(x_axis_subset)]        
    if add_column_dict:
        for k,v in add_column_dict.items():
            homo_scores_df[k]=v 
    homo_scores_df=homo_scores_df[homo_scores_df[y_axis].notna()]
    homo_scores_df=homo_scores_df.reset_index(drop=True)
    if replicates_grps:
        homo_scores_df =filt_homogeneity_score_df(homo_score_df=homo_scores_df,
                                                     replicates_cut_off=replicates_per_grp,
                                                     grps=replicates_grps)
    if split_groups:
        split_groups = list(set(split_groups[:]).intersection(homo_scores_df.columns.tolist()))
        grouped_df = homo_scores_df.groupby(split_groups)
        grps_dict = grouped_df.groups
        sample_keys=list(grps_dict.keys())
        for key in sample_keys:
            if fig_title is None:
                fig_title='_'.join(sorted([str(m) for m in key]))
                fig_title=''.join([fig_title,'_',y_axis+'_scores'])
            else:
                fig_title=re.sub(pattern='[\s\t]+',repl='_',string=fig_title)
            temp_homo_scores_df = homo_scores_df.iloc[grps_dict[key], :]
            temp_homo_scores_df=temp_homo_scores_df.reset_index(drop=True)
            #if  not temp_homo_scores_df : continue
            sorted_order = [cov for cov in col_order_value if cov in temp_homo_scores_df[col_order].unique()] \
            if col_order_value else [s for s in sorted(temp_homo_scores_df[col_order].unique())]
            #sorted_order = [s for s in sorted(temp_homo_scores_df[col_order].unique())]  
            if show_reps_counts:
                temp_homo_scores_dict=temp_homo_scores_df.groupby(show_reps_counts).groups
                for t_k in temp_homo_scores_dict.keys():
                    #if len(temp_homo_scores_dict[t_k])>=100:continue
                    #['fc','genes_count',hue]
                    print('fc:',temp_homo_scores_df.iloc[temp_homo_scores_dict[t_k],:].fc.unique())
                    print('genes_count:',temp_homo_scores_df.iloc[temp_homo_scores_dict[t_k],:].genes_count.unique())
                    print(hue,':',temp_homo_scores_df.iloc[temp_homo_scores_dict[t_k],:][hue].unique())
                    print('Reps per grp plotted:',len(temp_homo_scores_dict[t_k]))  
                    
            if limit:
                temp_homo_scores_dict=temp_homo_scores_df.groupby(['fc',hue,'genes_count']).groups
                filt_index=[]
                for t_k in temp_homo_scores_dict.keys():
                    if len(temp_homo_scores_dict[t_k])<=limit:
                        filt_index.extend(temp_homo_scores_dict[t_k])
                    else:
                        filt_index.extend(np.random.choice(temp_homo_scores_dict[t_k],size=limit,replace=False))
                        
                temp_homo_scores_df=temp_homo_scores_df.iloc[filt_index,:] 
            sns.set_style(style='white',rc={"xtick.major.size": 15, "ytick.major.size": 15})
            if in_kind=='bar':
                temp_facegrid = sns.catplot(x=x_axis, y=y_axis, hue=hue, col=col_order,col_wrap=in_col_wrap, 
                                        col_order=sorted_order,ci=ci,data=temp_homo_scores_df, kind=in_kind, 
                                        height=facet_size, aspect=1,hue_order=hue_order,palette=palette,
                                        sharex=sharex,sharey=sharey,capsize=capsize,n_boot=no_boots)
            elif in_kind=='box':
                temp_facegrid = sns.catplot(x=x_axis, y=y_axis, hue=hue, col=col_order,col_wrap=in_col_wrap, 
                                            col_order=sorted_order,ci=ci,data=temp_homo_scores_df, kind=in_kind,
                                            height=facet_size, aspect=1,hue_order=hue_order,palette=palette,
                                            sharex=sharex,sharey=sharey,n_boot=no_boots,
                                            **{'meanline':boxplot_mean,'showfliers':boxplot_outliers,
                                              'boxprops':boxprops})
            else:
                temp_facegrid = sns.catplot(x=x_axis, y=y_axis, hue=hue, col=col_order,col_wrap=in_col_wrap, 
                                            col_order=sorted_order,ci=ci,data=temp_homo_scores_df, kind=in_kind,
                                            height=facet_size, aspect=1,hue_order=hue_order,palette=palette,
                                            sharex=sharex,sharey=sharey,n_boot=no_boots)
            for ax in temp_facegrid.axes.flat:
                labels = ax.get_xticklabels() # get x labels
                ax.tick_params(length=6, width=.5,bottom=True,left=True)
                ax.set_xticklabels(labels, rotation=x_axis_rotation)
                #print(sns.color_palette())
            if add_swarm is True:
                for temp_ax in temp_facegrid.axes.flat:
                    #sns.set_style(style='white',rc={"xtick.major.size": 8, "ytick.major.size": 8})
                    ax_temp_homo_scores_df=\
                    temp_homo_scores_df[temp_homo_scores_df[col_order]==float(temp_ax.get_title().split('=')[-1])]
                    sns.swarmplot(ax=temp_ax,x=x_axis, y=y_axis,data=ax_temp_homo_scores_df,  
                                  hue=hue,hue_order=hue_order,dodge=True,palette=palette,size=swarm_points_size)
                    temp_ax.legend().set_visible(False)
            #temp_facegrid.set_xticklabels(rotation=70)   
            temp_facegrid.fig.suptitle(fig_title, size=subplot_title_size)            
            temp_facegrid.fig.subplots_adjust(top=subplot_adjust)  
            if out_dir:
                temp_dir=os.path.join(out_dir,y_axis)
                safe_mkdir(path=temp_dir)
                temp_out_fname = os.path.join(temp_dir, fig_title+'_'+in_kind+'.pdf')
                temp_facegrid.savefig(temp_out_fname)
    else:
        sorted_order = [s for s in sorted(homo_scores_df[col_order].unique())]
        temp_facegrid = sns.factorplot(x=x_axis, y=y_axis, hue=hue, col=col_order,
                                      col_wrap=in_col_wrap, col_order=sorted_order,ci=ci,
                                      data=homo_scores_df, kind=in_kind, size=4, aspect=1)
        temp_facegrid.set_xticklabels(rotation=70)        
        if out_dir:
            temp_dir=os.path.join(out_dir,y_axis)
            safe_mkdir(path=temp_dir)
            safe_mkdir(temp_dir)
            temp_out_fname = os.path.join(temp_dir, y_axis+'_score_'+in_kind+'.pdf')
            temp_facegrid.savefig(temp_out_fname)                
    pass

def filt_homogeneity_score_df(homo_score_df,grps=['method', 'cells_count', 'genes_count',
                                                  'fc', 'modification_method','clustering_method',
                                                  'randomize_cells_labels'],replicates_cut_off=10):
    '''Prints the homogeneity scores stats'''
    grps=list(set(grps).intersection(homo_score_df.columns))
    grouped_df=homo_score_df.groupby(grps)
    grps_dict=grouped_df.groups
    filt_df_list=[]
    for key in grps_dict.keys():        
        try :
            temp_homo_df=homo_score_df.loc[grps_dict[key],:]
            if temp_homo_df.shape[0]<replicates_cut_off : continue
            filt_df_list.append(temp_homo_df)            
        except ValueError:    
            pass
    if len(filt_df_list)>=1:
        filt_df=pd.concat(filt_df_list)
        filt_df=filt_df.reset_index(drop=True)
        return filt_df
    pass



def gene_parameters(indata):
    
    ''' Gets per gene parameters (mean, dispersion, non-detected genes proportion,etc) and stores in a dataframe'''
    
    gene_means=indata.mean(axis=1)
    
    gene_std=indata.std(axis=1)
    
    gene_vars=indata.var(axis=1)
    
    genes_cv2 = gene_vars / (gene_means ** 2)
    genes_cv = gene_std /gene_means
    
    genes_no_detected=indata[indata>0.0].count(axis=1)
    
    indata_dim=indata.shape
    
    genes_non_detected_proportion =(indata_dim[1]-genes_no_detected)/indata_dim[1]
    
    genes_params_df=pd.concat([genes_no_detected.to_frame('cells_counts'),gene_std.to_frame('std'),
                               gene_means.to_frame(name='mean_expr'),gene_vars.to_frame(name='var_expr'),
                               genes_cv2.to_frame(name='cv2'),
                               genes_non_detected_proportion.to_frame('non_expr_proportion'),genes_cv.to_frame('cv')],
                              axis=1)
    return genes_params_df

def remove_nonexpressedgenes(inputdata): 
    '''Filters genes with zero expression in all samples'''
    selected_gns=inputdata.index[inputdata[inputdata>0.0].count(axis=1)>=1]
    selected_samples=inputdata.columns[inputdata[inputdata>0.0].count(axis=0)>=1]
    output_data=inputdata.loc[selected_gns,selected_samples]
    return output_data

def remove_genes(inputdata, gene_pattern):
    '''Removes genes whose ids or names mateches the provided pattern. Good for removing e.g ERCCs'''
    pattern=re.compile(gene_pattern)
    genes_to_keep=[gn for gn in inputdata.index if pattern.search(string=gn)==None]
    outdata=inputdata.loc[genes_to_keep,:]
    return outdata

def genes_modification_subspace(infile_or_indata,outdir,preselected_genes=None,fc=[0.1,0.5,2,10],
                                summarized=True,cpus=10,write_output=True,genes_no=None):
    ''' Finds all genes with the same droupout rate after fold modification '''
    indata=None
    try:
        if os.path.exists(infile_or_indata):
            indata =pd.read_csv(infile_or_indata, sep='\t',header=0, index_col=0)
    except TypeError:
        indata=pd.DataFrame(infile_or_indata)
    indata=remove_nonexpressedgenes(inputdata=indata)
    gene_parameters_df=gene_parameters(indata)
    all_genes=gene_parameters_df.index.tolist()
    genes =[s for s in all_genes if s in preselected_genes] if preselected_genes else all_genes   
    args_list=[(gene_parameters_df,gn,f,genes_no) for gn in genes for f in fc]    
    pool=mp.Pool(processes=cpus)
    results=pool.starmap_async(genes_with_same_parameters,iterable=args_list).get()
    pool.close()
    pool.join()
    del args_list
    out_list=[]
    for result in results:
        matching_genes=result['matching_genes_list']
        matching_genes_counts=len(matching_genes)
        matching_genes=','.join(matching_genes) if matching_genes else 'nan'        
        if summarized==True:            
            out_list.append([matching_genes_counts,result['fc'],result['mean_interval'],result['gene']])
        else:
            out_list.append([matching_genes_counts, matching_genes,result['gene'],result['fc']])
        '''    
        else:
            out_list.append([matching_genes_counts,matching_genes,result['fc'],
                             result['mean_interval'],result['drop_out_interval'],
                             result['cv'],result['gene'],result['mean'],result['dropout']]) 
                             '''
    col_list=['matching_genes_count','fc','mean_interval','gene'] if summarized==True else ['matching_genes_count','genes','gene','fc']    
    out_df=pd.DataFrame(out_list,columns=col_list)
    if write_output:
        safe_mkdir(outdir)
        fc_str='_'.join([str(f) for f in  fc])
        out_fname=os.path.join(outdir,'fc_'+fc_str+'_gene_mod_subspace_summarized.tab') if summarized==True else os.path.join(outdir,'fc_'+fc_str+'_gene_mod_subspace.tab')
        write_mode = 'a' if os.path.exists(out_fname) == True else 'w'
        if write_mode == 'a':
            out_df.to_csv(path_or_buf=out_fname, sep='\t', header=False, index=True, mode=write_mode)
        else:
            out_df.to_csv(path_or_buf=out_fname, sep='\t', header=True, index=True, mode=write_mode)
        return out_fname
    else:
        return out_df
    pass

def plot_genes_with_same_parameter(genes_params_df,fig_out_fname,target_genes,fc=2,mean_interval=1.0,
                                   drop_out_interval=0.01,cv2=None,gene_with_same_parameter_summarized=False):
    '''Method to visualize genes with the same properties'''    
    if len(target_genes)>9:
        target_genes=np.random.choice(target_genes,9).tolist()
    fig = plt.figure(figsize=(15,15))
    for m  in range(len(target_genes)):
        target_gene=target_genes[m]        
        target_gene_mean=genes_params_df.loc[target_gene,'mean_expr']
        param_res_dict=genes_with_same_parameters(genes_params_df=genes_params_df, target_gene=target_gene,fc=fc,
                                                  mean_interval=mean_interval, drop_out_interval=drop_out_interval,
                                                  cv2=cv2)        
        genes=[target_gene]
        selected_genes_list=param_res_dict['matching_genes_list']
        if len(selected_genes_list)>=1:
            genes.extend(selected_genes_list)
            mean=np.log2(np.array(genes_params_df.loc[genes,'mean_expr'])+1)
            #mean = np.array(genes_params_df.loc[genes, 'mean_expr'])
            prop=genes_params_df.loc[genes,'non_expr_proportion']
            cols=['r' if g==target_gene else 'b' for g in genes]
            plt.subplot(3,3,m+1)
            plt.scatter(x=mean,y=prop,c=cols)
            title_str=' '.join([target_gene,'fc:',str(fc),'Cnts:',str(len(selected_genes_list)),'mn:',str(np.round(target_gene_mean,2))])
            plt.title(title_str)
            plt.xlabel('log2(Expr)')
            plt.ylabel('Proportion')
    fig.savefig(fig_out_fname)
    pass

def genes_with_same_parameters(genes_params_df,target_gene,fc=2.0,genes_no=None):    
    ''' Gets genes fitting the specific parameter interval in mean and drop-out rates '''    
    mean_series=genes_params_df.mean_expr
    dropout_series=genes_params_df.non_expr_proportion
    target_mean=np.float(mean_series[target_gene])
    #mean_interval=target_mean*0.1
    mean_interval=target_mean*fc*0.1
    mod_target_mean=target_mean*fc
    #mod_mean_interval=[(target_mean*fc)-mean_interval,(target_mean*fc)+mean_interval]
    mod_mean_interval=[(mod_target_mean)-mean_interval,(mod_target_mean)+mean_interval]
    matching_genes_parameters_df = genes_params_df[(genes_params_df.mean_expr >= mod_mean_interval[0]) &
                                              (genes_params_df.mean_expr <= mod_mean_interval[1])]
    matching_genes_list = matching_genes_parameters_df.index.tolist()
    matching_genes_list=np.random.choice( matching_genes_list,genes_no,replace=False).tolist() if not genes_no is None and len(matching_genes_list)>=genes_no else matching_genes_list
    out_dict={'matching_genes_list':matching_genes_list,'fc':fc,'mean_interval':mean_interval,
              'gene':target_gene,
              'mean':mean_series.loc[target_gene],'dropout':dropout_series[target_gene]}
    return out_dict

def genes_with_same_cv2_parameters(genes_params_df,target_gene,fc=2.0,genes_no=None):    
    ''' Gets genes fitting the specific parameter interval in cv2 and drop-out rates '''    
    cv_series=genes_params_df.cv2
    dropout_series=genes_params_df.non_expr_proportion
    target_cv=np.float(cv_series[target_gene])
    mean_cv_interval=target_cv*fc*0.1
    mod_cv_interval=[(target_cv*fc)-mean_cv_interval,(target_cv*fc)+mean_cv_interval]
    matching_genes_parameters_df = genes_params_df[(genes_params_df.cv2 >= mod_cv_interval[0]) &
                                              (genes_params_df.cv2 < mod_cv_interval[1])]
    matching_genes_list = matching_genes_parameters_df.index.tolist()
    matching_genes_list=np.random.choice( matching_genes_list,genes_no,replace=False).tolist() if not genes_no is None and len(matching_genes_list)>=genes_no else matching_genes_list
    
    out_dict={'matching_genes_list':matching_genes_list,'fc':fc,'cv_interval':mean_cv_interval,
              'gene':target_gene,
              'cv2':cv_series.loc[target_gene],'dropout':dropout_series[target_gene]}
    return out_dict

def filter_genes_dispersion(data,
                            flavor='seurat',
                            min_disp=None, max_disp=None,
                            min_mean=None, max_mean=None,
                            n_bins=20,
                            n_top_genes=None,
                            log=True,
                            copy=False):
    """Extract highly variable genes [Satija15]_ [Zheng17]_.
    If trying out parameters, pass the data matrix instead of AnnData.
    Depending on option `flavor`, this reproduces the R-implementations of
    Seurat [Satija15]_ and Cell Ranger [Zheng17]_.
    Use `flavor='cell_ranger'` with care and in the same way as in
    :func:`~scanpy.api.pp.recipe_zheng17`.
    Parameters
    ----------
    data : :class:`~scanpy.api.AnnData`, `np.ndarray`, `sp.sparse`
        Data matrix.
    flavor : {'seurat', 'cell_ranger'}, optional (default: 'seurat')
        Choose the flavor for computing normalized dispersion. If choosing
        'seurat', this expects non-logarithmized data - the logarithm of mean
        and dispersion is taken internally when `log` is at its default value
        `True`. For 'cell_ranger', this is usually called for logarithmized data
        - in this case you should set `log` to `False`. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
    min_mean=0.0125, max_mean=3, min_disp=0.5, max_disp=`None` : `float`, optional
        If `n_top_genes` is not `None`, these cutoffs for the normalized gene
        expression are ignored.
    n_bins : `int` (default: 20)
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    n_top_genes : `int` or `None` (default: `None`)
        Number of highly-variable genes to keep.
    log : `bool`, optional (default: True)
        Use the logarithm of the mean to variance ratio.
    copy : `bool`, optional (default: `False`)
        If an AnnData is passed, determines whether a copy is returned.
    Returns
    -------
    If an AnnData `adata` is passed, returns or updates `adata` depending on \
    `copy`. It filters the `adata` and adds the annotations
    means : adata.var
        Means per gene. Logarithmized when `log` is `True`.
    dispersions : adata.var
        Dispersions per gene. Logarithmized when `log` is `True`.
    dispersions_norm : adata.var
        Normalized dispersions per gene. Logarithmized when `log` is `True`.
    If a data matrix `X` is passed, the annotation is returned as `np.recarray` \
    with the columns: `gene_subset`, `means`, `dispersions`, `dispersion_norm`.
    """
    if n_top_genes is not None and not all([
            min_disp is None, max_disp is None, min_mean is None, max_mean is None]):
        pass
    if min_disp is None: min_disp = 0.5
    if min_mean is None: min_mean = 0.0125
    if max_mean is None: max_mean = 3
    #print('scanpy: find variable genes\n')
    X = data.transpose()  # no copy necessary, X remains unchanged in the following
    mean, var = _get_mean_var(X)
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    if log:  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    # all of the following quantities are "per-gene" here
    import pandas as pd
    df = pd.DataFrame()
    df['mean'] = mean
    df['dispersion'] = dispersion
    if flavor == 'seurat':
        df['mean_bin'] = pd.cut(df['mean'], bins=n_bins)
        disp_grouped = df.groupby('mean_bin')['dispersion']
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()
        gen_indices = np.where(one_gene_per_bin[df['mean_bin']])[0].tolist()
        if len(gen_indices) > 0:
            '''
            print(
                'Gene indices {} fell into a single bin: their '
                'normalized dispersion was set to 1.\n    '
                'Decreasing `n_bins` will likely avoid this effect.'
                .format(gen_indices))
                '''
        disp_std_bin[one_gene_per_bin] = disp_mean_bin[one_gene_per_bin]
        disp_mean_bin[one_gene_per_bin] = 0
        # actually do the normalization
        df['dispersion_norm'] = (df['dispersion'].values  # use values here as index differs
                                 - disp_mean_bin[df['mean_bin']].values) \
                                 / disp_std_bin[df['mean_bin']].values
    elif flavor == 'cell_ranger':
        from statsmodels import robust
        df['mean_bin'] = pd.cut(df['mean'], np.r_[-np.inf,
            np.percentile(df['mean'], np.arange(10, 105, 5)), np.inf])
        disp_grouped = df.groupby('mean_bin')['dispersion']
        disp_median_bin = disp_grouped.median()
        # the next line raises the warning: "Mean of empty slice"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            disp_mad_bin = disp_grouped.apply(robust.mad)
        df['dispersion_norm'] = np.abs((df['dispersion'].values
                                 - disp_median_bin[df['mean_bin']].values)) \
                                / disp_mad_bin[df['mean_bin']].values
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    dispersion_norm = df['dispersion_norm'].values.astype('float32')
    if n_top_genes is not None:
        dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
        disp_cut_off = dispersion_norm[n_top_genes-1]
        gene_subset = df['dispersion_norm'].values >= disp_cut_off
        
    else:
        max_disp = np.inf if max_disp is None else max_disp
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce((mean > min_mean, mean < max_mean,
                                             dispersion_norm > min_disp,
                                             dispersion_norm < max_disp))
    #print ("scanpy: mvg finished (%i)" % sum(gene_subset))
    return np.rec.fromarrays((gene_subset,
                              df['mean'].values,
                              df['dispersion'].values,
                              df['dispersion_norm'].values.astype('float32', copy=False)),
                              dtype=[('gene_subset', bool),
                                     ('means', 'float32'),
                                     ('dispersions', 'float32'),
                                     ('dispersions_norm', 'float32')])
def _get_mean_var(X):
    # - using sklearn.StandardScaler throws an error related to
    #   int to long trafo for very large matrices
    # - using X.multiply is slower
    if True:
        mean = X.mean(axis=0)
        if issparse(X):
            mean_sq = X.multiply(X).mean(axis=0)
            mean = mean.A1
            mean_sq = mean_sq.A1
        else:
            mean_sq = np.multiply(X, X).mean(axis=0)
        # enforece R convention (unbiased estimator) for variance
        var = (mean_sq - mean**2) * (X.shape[0]/(X.shape[0]-1))
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False).partial_fit(X)
        mean = scaler.mean_
        # enforce R convention (unbiased estimator)
        var = scaler.var_ * (X.shape[0]/(X.shape[0]-1))
    return mean, var


def plot_genes_modification_subspace_distribution(subspace_infile, output_dir, x_axis='quantiles',
                                                  y_axis='matching_genes_count',in_kind='point',
                                                  in_col_wrap=3, hue='fc', col_order='fc',split_groups=None):
    '''Plots the matching genes distributions'''
    safe_mkdir(output_dir)
    subspace_df = pd.read_csv(subspace_infile, sep='\t', header=0)    
    if split_groups:
        split_groups=set(split_groups).intersection(subspace_df.columns)
        grouped_df=subspace_df.groupby(split_groups)
        grps_dict = grouped_df.groups
        for key in grps_dict.keys():
            temp_subspace_df = subspace_df.loc[grps_dict[key], :]
            sorted_order = [s for s in sorted(temp_subspace_df[col_order].unique())]             
            temp_facegrid = sns.factorplot(x=x_axis, y=y_axis, hue=hue, col=col_order,
                                          col_wrap=in_col_wrap, col_order=sorted_order,
                                          data=temp_subspace_df, kind=in_kind)
            temp_facegrid.set_xticklabels(rotation=70)
            group='_'.join([str(k) for k in key])
            temp_out_fname = os.path.join(output_dir, group + '_gene_modification_subspace.pdf')
            temp_facegrid.savefig(temp_out_fname)
    else:
        sorted_order = [s for s in sorted(subspace_df[col_order].unique())]         
        temp_facegrid = sns.factorplot(x=x_axis, y=y_axis, hue=hue, col=col_order,
                                      col_wrap=in_col_wrap, col_order=sorted_order,
                                      data=subspace_df, kind=in_kind)
        temp_facegrid.set_xticklabels(rotation=70)
        temp_out_fname = os.path.join(output_dir, 'gene_modification_subspace.pdf')
        temp_facegrid.savefig(temp_out_fname)
    pass

def plot_heatmap(data_frame,title='Heatmap',xlab='X-axis',ylab='Y-axis',labs_font_size=10,
                 xlabs_ratotation=70,ylabs_ratotation=0,cmap='seismic'):
    ax_heat = sns.heatmap(data=data_frame,cmap=cmap)
    ax_heat.set(xlabel=xlab, ylabel=ylab)
    x_tick_labs=ax_heat.get_xticklabels()
    y_tick_labs = ax_heat.get_yticklabels()
    ax_heat.set_xticklabels(x_tick_labs, rotation=xlabs_ratotation, fontsize=labs_font_size)
    ax_heat.set_yticklabels(y_tick_labs,rotation=ylabs_ratotation, fontsize=labs_font_size)
    return  ax_heat

#Basic utility methods for running the code

def safe_mkdir(path):
    '''Makes directory in safe manner and if the intermediary folders are missing, creates them too'''    
    if not os.path.exists(path):
        os.makedirs(path,mode=0o777);
    else:     
        pass
    pass

def get_all_files_in_directory(directory):
    '''Returns a list of all files plus their full directory path's  in root plus the intermediate directories '''
    all_dirs = []
    all_files = []
    for (root, dirs, filenames) in os.walk(directory):
        all_dirs.append(root)
    for d in all_dirs:
        [all_files.append(d + '/' + f) for f in os.listdir(d)]
    return all_files

def get_list_of_specific_file_type(directory, file_pattern):
    '''Returns a list of certain file types given a directory and a pattern matching the file e.g its extension'''
    all_files = get_all_files_in_directory(directory);    
    pattern=re.compile(file_pattern)
    specific_file_list = [];
    for current_file in all_files:
        if pattern.search(string=current_file)==None : continue
        specific_file_list.append(current_file)
    return specific_file_list


def fano_variable(DGE,input_mean=None,meanthresh=0.5,resthresh=0.05,f=0.25,highlight_genes=None,plot=0):
    #get mean and std for each gene
    if input_mean is None:
        popmean=np.log2(np.mean(DGE,axis=1)+1)
    else:
        popmean=input_mean        
    popstd=np.std(np.log2(DGE+1),axis=1)#np.divide(np.std(DGE,axis=1),popmean)
    thresh=meanthresh
    x=popmean[np.array(popmean>thresh)]
    y=popstd[np.array(popmean>thresh)]
    DGE_fit=DGE[np.array(popmean>thresh)]
    #fit line
    lowess = sm.nonparametric.lowess
    lz_pred = lowess(y, x,frac=f,return_sorted=False)
    residuals=y-lz_pred
    if plot==1:
        plt.scatter(x,y,c=['red' if z>resthresh else 'blue' for z in residuals])
        plt.xlabel('log2(Population Mean)')
        plt.ylabel('Standard Deviation')
    df_res=pd.DataFrame()
    df_res['residuals']=residuals
    df_res['mean']=x
    df_res['std']=y
    df_res['sig']=[True if z>resthresh else False for z in residuals]
    df_res.index=DGE_fit.index
    if highlight_genes:
        if plot==1:
            subset=df_res.loc[highlight_genes].dropna()
            for thisgene in subset.index:
                df_tmp=subset.loc[thisgene]
                plt.text(df_tmp['mean'],df_tmp['std'],thisgene,fontsize=16)
    return df_res

#Temporal methods to test
def process_data_in_scanpy(in_adata,min_cells=3,min_genes=1,regress_grps=None,top_var_gns=None,no_pcs=10,
                 tsne_perplexity=5,log=None,normalize_per_cell=None,scale=None,plot=None):
    scanpi.pp.filter_genes(data=in_adata, min_cells=min_cells)
    scanpi.pp.filter_cells(data=in_adata, min_genes=min_genes)
    if normalize_per_cell:
        scanpi.pp.normalize_per_cell(in_adata,counts_per_cell_after=500000)
    if log:
        scanpi.pp.log1p(data=in_adata)
    if scale:
        scanpi.pp.scale(in_adata)
    if regress_grps:
        scanpi.pp.regress_out(adata=in_adata, keys=regress_grps)
    if top_var_gns:
        scanpi.pp.filter_genes_dispersion(in_adata, n_top_genes=top_var_gns)
    if plot:
        scanpi.tl.pca(data=in_adata,n_comps=no_pcs)
        scanpi.tl.tsne(adata=in_adata,n_pcs=no_pcs,perplexity=tsne_perplexity)
        scanpi.pp.neighbors(adata=in_adata)
        scanpi.tl.louvain(adata=in_adata)
        scanpi.tl.umap(adata=in_adata,n_components=no_pcs)
    return in_adata


def fb_pca(DGE,k=50):
    pca=PCA(n_components=k)
    pca.fit(DGE)
    Ufb=pca.fit_transform(DGE)
    Sfb=pca.explained_variance_
    Vfb=pca.components_
    Vfb=pd.DataFrame(Vfb).T
    Vfb.index=DGE.columns
    Ufb=pd.DataFrame(Ufb)
    Ufb.index=DGE.index
    return Ufb,Sfb,Vfb


def permute_matrix(DGE,bins=20,verbose=0):
    """Permute genes based on similar expression levels"""
    DGE_perm=DGE.copy()
    GSUMS=np.sum(DGE,axis=1)
    breakvec = np.linspace(1,100,bins)
    breaks=[]
    for breaker in breakvec:
        breaks.append(np.percentile(GSUMS,breaker))
    breaks=np.unique(breaks)
    for i in range(len(breaks)-1):
        if verbose==1:
            print(np.round((1.0*i)/(len(breaks)-1)))
        for j in range(len(DGE.columns)):
            curlogical=np.logical_and(GSUMS>breaks[i],GSUMS<=breaks[i+1])
            DGE_perm.ix[curlogical,j]=np.random.permutation(DGE_perm.ix[curlogical,j])
    return DGE_perm


def jackstraw(DGEZ,per=0.005,sig_pcs=40,reps=100,verbose=0):
    """substitute small percentage of features with permuted versions, compare actual to permuted to obtain significance"""
    ngenes=len(DGEZ)
    [Ufb,Sfb,Vfb]= fb_pca(DGEZ,k=sig_pcs)
    Ufb_null = pd.DataFrame()
    flag=0
    #repeatedly permute and recalculate null PC distributions
    for i in range(reps):
        if (verbose==1):
            print('rep',i)
        shuf_genes=np.random.choice(range(ngenes),size=int(np.ceil(ngenes*per)),replace=False)
        DGEZ_perm=DGEZ.copy()
        DGEZ_perm.ix[shuf_genes,:]=np.array(DGEZ_perm.ix[shuf_genes,np.random.permutation(range(np.shape(DGEZ)[1]))])
        #[Ufb_perm,Sfb_perm,Vfb_perm]= fb_pca(DGEZ,k=sig_pcs)
        [Ufb_perm,Sfb_perm,Vfb_perm]= fb_pca(DGEZ_perm,k=sig_pcs)
        #tmp_null=Ufb.ix[shuf_genes,:]
        tmp_null=Ufb_perm.ix[shuf_genes,:]
        if flag==0:
            Ufb_null=tmp_null
            flag=1
        else:
            Ufb_null=pd.concat([Ufb_null,tmp_null])
    PVALS=Ufb.copy()
    for i in range(sig_pcs):
        curecdf=ECDF(Ufb_null.ix[:,i])
        curUfb=Ufb.ix[:,i]
        isnegative=curUfb<0.0
        ispositive=curUfb>=0.0
#statsmodels.sandbox.stats.multicomp.fdrcorrection0
        PVALS.ix[isnegative,i]=np.log10(curecdf(Ufb.ix[isnegative,i]))
        PVALS.ix[ispositive,i]=-np.log10(1-curecdf(Ufb.ix[ispositive,i]))
        PVALS[PVALS>5]=5
        PVALS[PVALS<(-5)]=-5
    return PVALS


def fano_variable(DGE,input_mean=None,meanthresh=0.5,resthresh=0.05,f=0.25,
                  highlight_genes=None,plot=0):
    #get mean and std for each gene
    if input_mean is None:
        popmean=np.log2(np.mean(DGE,axis=1)+1)
    else:
        popmean=input_mean        
    popstd=np.std(np.log2(DGE+1),axis=1)#np.divide(np.std(DGE,axis=1),popmean)
    thresh=meanthresh
    x=popmean[np.array(popmean>thresh)]
    y=popstd[np.array(popmean>thresh)]
    DGE_fit=DGE[np.array(popmean>thresh)]
    #fit line
    lowess = sm.nonparametric.lowess
    lz_pred = lowess(y, x,frac=f,return_sorted=False)
    residuals=y-lz_pred
    if plot==1:
        plt.scatter(x,y,c=['red' if z>resthresh else 'blue' for z in residuals])
        plt.xlabel('log2(Population Mean)')
        plt.ylabel('Standard Deviation')
    df_res=pd.DataFrame()
    df_res['residuals']=residuals
    df_res['mean']=x
    df_res['std']=y
    df_res['sig']=[True if z>resthresh else False for z in residuals]
    df_res.index=DGE_fit.index
    if highlight_genes:
        if plot==1:
            subset=df_res.loc[highlight_genes].dropna()
            for thisgene in subset.index:
                df_tmp=subset.loc[thisgene]
                plt.text(df_tmp['mean'],df_tmp['std'],thisgene,fontsize=16)
    return df_res
#return variable genes

def PC_noise(DGEZ,noiselevels=np.linspace(-2,2,20),reps=3,sig_pcs=40):
    PC_cor=pd.DataFrame()
    [Ufb,Sfb,Vfb]= fb_pca(DGEZ,k=sig_pcs)
    for noise in noiselevels:
        df_noise=pd.DataFrame()
        for rep in range(reps):
            DGE_Z_wnoise=DGEZ+np.random.normal(0,np.power(10.0,noise),np.shape(DGEZ))
            [Ufb_noise,Sfb_noise,Vfb_noise]=fb_pca(DGE_Z_wnoise,k=sig_pcs)
            comparevec=[]
            for i in range(sig_pcs):
                corrs=[]
                for j in range(sig_pcs):
                    corrs.append(np.abs(np.corrcoef(Ufb.ix[:,j],Ufb_noise.ix[:,i])[0][1]))
                comparevec.append(np.max(corrs))
            df_noise[rep]=comparevec
        PC_cor[noise]=df_noise.mean(axis=1)
    return PC_cor

def test_stats_btw_grps(scores_infile,score='hs',out_dir=None,x_axis='components',replicates_per_grp=40,
                split_groups=None,col_order='fc',in_row=None,x_axis_subset=None,hue='category',
                replicates_grps=None,in_kind='box',in_col_wrap=4,ci=95,col_subset=None,hue_order=None,
                fc_subset=None,gn_counts_subset=None,grps_comparisons='protocol',palette='bright',
                        x_axis_rotation=0,sharey=True,sharex=True):
    '''Compare statistics'''
    scores_df=None
    expt_name=None
    try:
        if os.path.exists(scores_infile):
            expt_name = scores_infile.split('/')[-2]
            scores_df = pd.read_csv(scores_infile, sep='\t', header=0)            
    except TypeError:
        scores_df=pd.DataFrame(scores_infile)
    if col_subset:        
            scores_df=scores_df[scores_df[col_order].isin(col_subset)]
    if fc_subset:        
            scores_df=scores_df[scores_df['fc'].isin(fc_subset)]
    if gn_counts_subset:
        scores_df=scores_df[scores_df['genes_count'].isin(gn_counts_subset)]    
    
    scores_df=scores_df.astype({score:str})
    scores_df=scores_df[scores_df[score]!='nan']
    scores_df=scores_df.astype({score:np.float})
    scores_df =filt_homogeneity_score_df(homo_score_df=scores_df,
                                         replicates_cut_off=replicates_per_grp,
                                         grps=replicates_grps)
    if scores_df is None: 
        print('No replicates passed the required cut-off of: ',replicates_per_grp)
        return     
    else:
        all_comparison_grps=list(scores_df[grps_comparisons].unique())
        all_comparison_grps_comparisons=[[p[0],p[1]] for p in itertools.combinations(all_comparison_grps,2)] 
        if split_groups:
            out_list=[]
            stats_lst=[]
            split_groups = list(set(split_groups).intersection(scores_df.columns))
            grouped_df = scores_df.groupby(split_groups)
            grps_dict = grouped_df.groups
            for ky,v in grps_dict.items():
                temp_scores_df=scores_df.loc[v,:]
                for temp_comp in all_comparison_grps_comparisons:
                    f_temp_scores_df=temp_scores_df[temp_scores_df[grps_comparisons]==temp_comp[0]]
                    s_temp_scores_df=temp_scores_df[temp_scores_df[grps_comparisons]==temp_comp[1]]
                    f_tmp_vals=f_temp_scores_df[score].values
                    s_tmp_vals=s_temp_scores_df[score].values
                    f_temp_stats_res=stats.describe(f_tmp_vals)
                    s_temp_stats_res=stats.describe(s_tmp_vals)                    
                    print(len(f_tmp_vals))                    
                    print(len(s_tmp_vals))                    
                    temp_t_test_stat,temp_t_test_pval =ttest_ind(f_tmp_vals,s_tmp_vals, nan_policy='omit',
                                                                 equal_var=False)
                    fc=''.join([str(f) for f in temp_scores_df['fc'].unique().tolist()])
                    gn_ctns=''.join([str(f) for f in temp_scores_df['genes_count'].unique().tolist()])
                    cells_cnts=''.join([str(f) for f in temp_scores_df['cells_count'].unique().tolist()])
                    cat=fc+'_'+gn_ctns+'_'+cells_cnts
                    temp_sig='sig' if temp_t_test_pval < 0.01 else 'ns'
                    grp_cat=temp_comp[0]+'_vs_'+temp_comp[1]
                    stats_lst.append([grp_cat,temp_comp[0],temp_comp[1],fc,gn_ctns,cells_cnts,f_temp_stats_res[2],
                                      s_temp_stats_res[2],np.median(f_tmp_vals),np.median(s_tmp_vals),
                                      f_temp_stats_res[3],s_temp_stats_res[3],temp_t_test_pval,temp_sig])                    
                    out_list.extend([[f_v,fc,gn_ctns,cat,temp_comp[0],grp_cat,temp_sig] for f_v in f_tmp_vals])
                    out_list.extend([[s_v,fc,gn_ctns,cat,temp_comp[1],grp_cat,temp_sig] for s_v in s_tmp_vals])  
            stats_df=pd.DataFrame(stats_lst,columns=['comparison', 'grp_one', 'grp_two', 'fc', 'genes_cnts','cells_count', 
                                                     'grp_one_mean_'+score, 'grp_two_mean_'+score, 'grp_one_median_'+score, 
                                                     'grp_two_median_'+score, 'grp_one_variance_'+score, 
                                                     'grp_two_variance_'+score, 
                                                     'p_value', 'significance'])            
            out_df=pd.DataFrame(out_list,columns=[score,'fc','gn_cnts','category',grps_comparisons,'grp','sig'])
            all_grps=out_df.category.unique()
            all_grps_dict=dict(zip(all_grps,[m+1 for m in range(len(all_grps))]))
            out_df['x_axis']=[all_grps_dict[cat] for cat in out_df.category]
            fig=sns.factorplot(data=out_df,kind=in_kind,col='grp',row=in_row,hue=grps_comparisons,sharex=sharex,
                               sharey=sharey,x='category',palette=palette,y=score,col_wrap=in_col_wrap) 
            fig.set_xticklabels(rotation=x_axis_rotation)
            if out_dir:
                out_dir=out_dir+'/'+score
                safe_mkdir(out_dir)
                fig.savefig(os.path.join(out_dir,score+'_mean_score.pdf')) 
                stats_df.to_csv(os.path.join(out_dir,score+'_mean_score.tab'),sep='\t')                       
        else:
            print('To-do...')                  
    pass

def process_data_in_scanpy(in_adata,min_cells=3,min_genes=1,regress_grps=None,top_var_gns=None,no_pcs=10,
                 tsne_perplexity=5,log=None,normalize_per_cell=None,scale=None,plot=None):
    scanpi.pp.filter_genes(data=in_adata, min_cells=min_cells)
    scanpi.pp.filter_cells(data=in_adata, min_genes=min_genes)
    if normalize_per_cell:
        scanpi.pp.normalize_per_cell(in_adata,counts_per_cell_after=500000)
    if log:
        scanpi.pp.log1p(data=in_adata)
    if scale:
        scanpi.pp.scale(in_adata)
    if regress_grps:
        scanpi.pp.regress_out(adata=in_adata, keys=regress_grps)
    if top_var_gns:
        scanpi.pp.filter_genes_dispersion(in_adata, n_top_genes=top_var_gns)
    if plot:
        scanpi.tl.pca(data=in_adata,n_comps=no_pcs)
        scanpi.tl.tsne(adata=in_adata,n_pcs=no_pcs,perplexity=tsne_perplexity)
        scanpi.pp.neighbors(adata=in_adata)
        scanpi.tl.louvain(adata=in_adata)
        scanpi.tl.umap(adata=in_adata,n_components=no_pcs)
    return in_adata
