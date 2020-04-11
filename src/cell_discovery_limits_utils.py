import pandas as pd, celldiscoveryutilities as utils, seaborn as sns
import subprocess
from joblib import Parallel, delayed
import os,copy,pickle,re,time,itertools,random,numpy as np
import statsmodels.api as sm,math as ma, pandas as pd
from sklearn import  metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import neighbors
from scipy.stats import poisson
from numpy.random import beta, poisson
import multiprocessing as mp
from scipy.stats import binom
from sklearn.manifold import TSNE
from sklearn.utils.validation import check_array
import signal
from matplotlib import pyplot as pl
from matplotlib import rcParams
from scipy.sparse import issparse
from scipy.stats import levene
import hdbscan
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from scanpy.api.pp import filter_genes_dispersion
import scanpy.api as scanpi


## Step 1. Functions for reading,transformation and perturbing gene expression dataframe
def read_inputdata(filepath,sep='\t',header=0,index_col=0,nrows=None):
    
    '''Reads data table into a data frame'''
    
    with open(filepath,'r') as infh:
        
        input_df=pd.read_csv(filepath,sep=sep,header=header,index_col=index_col,nrows=nrows)
        return (input_df)
    
def prepare_data(inputdata, fc, no_genes,  no_cells, method='multiplicative',quantiles=None,
                 filter_genes_pattern=None,genes_and_cells_cut_off=None,normalize=None,
                 pseudo_value=1,find_var_genes=None,preselected_genes=None,
                 gene_modification_profile_df=None,gene_parameters_df=None,genes_in_all_cells=False,
                 match_ranks=None,ranks_to_select=None):
    
    data = remove_nonexpressedgenes(inputdata)
    
    if filter_genes_pattern:
        #Useful in cases where user may want to exclude e.g ERCC
        data = remove_genes(inputdata=data, gene_pattern=filter_genes_pattern)
    if normalize:
        #In case you provide counts instead of normalized expression
        norm_data_dict=utils.get_genes_sf(counts_df=data,pseudo_value=pseudo_value,loc_func=np.median,
                                          return_norm_expr=True)
        data =norm_data_dict['norm']
    if quantiles:
        #Get only genes from a specific mean quantile interval
        data=genes_quantiles(data,lower_quantile=quantiles[0],upper_quantile=quantiles[1]) 
    if genes_and_cells_cut_off:
        #Gets genes expressed in a specific number of cells
        data=filter_genes_and_samples(data,cut_off=genes_and_cells_cut_off[0],
                                      no_samples=genes_and_cells_cut_off[1],
                                      genes_count_cut_off=genes_and_cells_cut_off[2])
        
    perturbed_data =perturb_data(inputdata=data,fc=fc,no_genes=no_genes,no_cells=no_cells,
                                 method=method,preselected_genes=preselected_genes,
                                 genes_in_all_cells=genes_in_all_cells,
                                 gene_modification_profile_df=gene_modification_profile_df,
                                 gene_parameters_df=gene_parameters_df,match_ranks=match_ranks,
                                 ranks_to_select=ranks_to_select)
    
    return perturbed_data

def perturb_data(inputdata,fc,no_genes,no_cells,method='multiplicative',preselected_genes=None,
                 genes_in_all_cells=False,gene_modification_profile_df=None,gene_parameters_df=None,
                 match_ranks=None,ranks_to_select=None):
    '''Modifys expression data frame of a given set of genes, in target cells using the input method''' 
    all_cells=inputdata.columns.tolist()
    #np.random.seed(seed=115678)
    cells=np.random.choice(all_cells,no_cells,replace=False)
    all_genes_set=set(inputdata.index.tolist())
    #np.random.seed(seed=None)
    genes_to_select_from_set=all_genes_set
    if preselected_genes:
        genes_to_select_from_set=genes_to_select_from_set.intersection(preselected_genes)
    
    if genes_in_all_cells == True:
        temp_target_cell_df=inputdata.loc[:,cells]
        temp_target_cells_genes_list=temp_target_cell_df.index[temp_target_cell_df[temp_target_cell_df>0.0].count(axis=1)
                                                            ==no_cells].tolist()
        
        genes_to_select_from_set=genes_to_select_from_set.intersection(temp_target_cells_genes_list)
    try:
        out_dict = {}
        if (method == 'multiplicative'):
            genes = np.random.choice(list(genes_to_select_from_set), size=no_genes, replace=False)
            out_dict['genes']= genes
            out_dict['cells']= cells
            out_dict['fc']= fc
            mod_data = multiplicative_modification(inputdata=inputdata, cells=cells, genes=genes, fc=fc)
            out_dict['data']=mod_data
        if (method == 'mean'):
            genes = np.random.choice(list(genes_to_select_from_set), size=no_genes, replace=False)
            out_dict['genes'] = genes
            out_dict['cells'] = cells
            out_dict['fc'] = fc
            mod_data = mean_modification(inputdata=inputdata, genes=genes, cells=cells, fc=fc)
            out_dict['data']=mod_data
        if (method == 'synthetic'):
            genes = np.random.choice(list(genes_to_select_from_set), size=no_genes, replace=False)
            out_dict['genes'] = genes
            out_dict['cells'] = cells
            out_dict['fc'] = fc
            mod_data = add_synthetic_genes_modification(inputdata=inputdata, genes=genes, cells=cells, fc=fc)
            out_dict['data']=mod_data
        if (method=='multiplicative_modification_space'):
            mod_dict=modification_in_subspace(inputdata=inputdata,cells=cells,no_genes=no_genes,fc=fc,
                                              gene_parameters_df=gene_parameters_df,all_genes_set=all_genes_set,
                                              genes_to_select_from_set=genes_to_select_from_set,
                                              gene_modification_profile_df=gene_modification_profile_df,
                                             match_ranks=match_ranks,ranks_to_select=ranks_to_select)
            out_dict['cells'] = cells
            out_dict['fc'] = fc
            out_dict['data'] = mod_dict['data']
            out_dict['genes'] = mod_dict['genes']
            out_dict['matching_genes'] = mod_dict['matching_genes']
            out_dict['ranks'] = mod_dict['ranks']
            
        if (method=='add_marker_gene_modification'):
            mod_dict=add_marker_gene_modification(inputdata=inputdata,cells=cells,no_genes=no_genes,fc=fc,
                                              gene_parameters_df=gene_parameters_df,all_genes_set=all_genes_set,
                                              genes_to_select_from_set=genes_to_select_from_set,
                                              gene_modification_profile_df=gene_modification_profile_df)
            out_dict['cells'] = cells
            out_dict['fc'] = fc
            out_dict['data'] = mod_dict['data']
            out_dict['genes'] = mod_dict['genes']
            out_dict['matching_genes'] = mod_dict['matching_genes'] 
        return out_dict
    
    
    except KeyboardInterrupt as key_err:
        print(key_err, 'thrown!!!', 'keyboard interruption during modification')
        
        pass
    except ValueError as val_err:
        print(val_err,'ValueError error captured during the modification')
        pass
    except IndexError as ind_error:
        print(ind_error,'IndexError error captured during modification')
        
        pass
    
    pass

def genes_quantiles(inputdata,lower_quantile=0.0,upper_quantile=1.0):
    
    # returns the input data frame after selecting gene from the specific mean-quantiles 
    
    mean_series=inputdata.mean(axis=1)
    lower_cut_off=mean_series.quantile(np.float(lower_quantile))
    upper_cut_off = mean_series.quantile(np.float(upper_quantile))
    
    mean_series_filt=mean_series[mean_series>=lower_cut_off]
    
    if np.round(upper_quantile,2)<1.00:
        
        mean_series_filt=mean_series_filt[mean_series_filt < upper_cut_off]  
        
    else:
        mean_series_filt[mean_series_filt <= upper_cut_off]
        
    output_data=inputdata.loc[mean_series_filt.index,inputdata.columns]
    return output_data

def multiplicative_modification(inputdata,genes,cells,fc=2,mean_interval=2.0,drop_out_interval=0.1,cv2=None):
    '''Given specific samples and genes, modifys the expression levels based on the provided fold change'''
    inputdata.loc[genes, cells]= inputdata.loc[genes, cells].mul(fc)
    return inputdata

def modification_in_subspace(inputdata,cells,no_genes,fc,gene_parameters_df,all_genes_set=None,genes_to_select_from_set=None,
                            gene_modification_profile_df=None,match_ranks=None,ranks_to_select=None):
    '''Given specific cells and number of genes, modifys the expression levels based on the provided fold change and
    ensures the mean is within a target subspace '''
    all_genes_set=set(inputdata.index.tolist()) if all_genes_set is None else all_genes_set.intersection(inputdata.index.tolist())
    profiled_fc=None if gene_modification_profile_df is None else gene_modification_profile_df.fc.unique().tolist()
    genes_to_select_from_set=all_genes_set if genes_to_select_from_set is None else genes_to_select_from_set.intersection(all_genes_set)
    gene_parameters_df=gene_parameters_df.loc[all_genes_set,:]
    all_genes_set_as_list=list(all_genes_set)
    len_cells=len(cells)
    all_cells=inputdata.columns.tolist()     
    genes=[]      
    matching_genes=[]
    ranks=[]
    matching_genes_dict={}
    out_dict={}
    matched_cells=np.random.choice(all_cells,len_cells,replace=False)
    genes_to_select_from_temp_params=gene_parameters_df.loc[genes_to_select_from_set,:]
    genes_to_select_from_ranked=pd.DataFrame([pd.Series.sort_values(genes_to_select_from_temp_params['mean_expr'])]).transpose()
    genes_to_select_from_ranked['rank']=[r for r in range(genes_to_select_from_ranked.shape[0])]
    if fc==0.0:
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        selected_genes=np.random.choice(genes_to_select_from_temp_list,no_genes,replace=False).tolist()
        out_dict['genes'] = selected_genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=selected_genes
        ranks.append(genes_to_select_from_ranked.loc[selected_genes,'rank'])
    elif profiled_fc is None or fc not in profiled_fc:
        mean_series = gene_parameters_df.mean_expr
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        len_genes_to_select_from_temp_list=len(genes_to_select_from_temp_list)
        len_limit=no_genes if no_genes<=len_genes_to_select_from_temp_list else len_genes_to_select_from_temp_list
        checked_genes = []
        matched_genes_tracker=set()
        if match_ranks is None:
            while len(matching_genes_dict)<len_limit:
                temp_gns_to_check=list(genes_to_select_from_set.difference(checked_genes))
                if len(temp_gns_to_check)==0:break
                temp_gn=np.random.choice(temp_gns_to_check, 1)[0]
                checked_genes.append(temp_gn)
                target_mean = np.float(mean_series.loc[temp_gn])*fc
                interval=0.1*target_mean
                mod_mean_interval = [target_mean - interval, target_mean+ interval]
                temp_matching_genes_list = gene_parameters_df[(gene_parameters_df.mean_expr >= mod_mean_interval[0]) &
                                                                  (gene_parameters_df.mean_expr <= mod_mean_interval[1])].index.tolist()
                if not temp_matching_genes_list: continue
                matching_genes_set=all_genes_set.intersection(temp_matching_genes_list)
                if not matching_genes_set : continue
                diff_matching_genes_set=matching_genes_set.difference(matched_genes_tracker)
                tmp_matching_gene=np.random.choice(list(matching_genes_set),1)[0] if diff_matching_genes_set else np.random.choice(list(matching_genes_set),1)[0]
                genes.append(temp_gn)
                matching_genes.append(tmp_matching_gene)
                matching_genes_dict[temp_gn]=tmp_matching_gene
                matched_genes_tracker.add(tmp_matching_gene)
                ranks.append(genes_to_select_from_ranked.loc[temp_gn,'rank'])
        #del matched_genes_tracker
        else:
            ranks_to_select_copy=ranks_to_select[:]
            temp_gns_to_check=genes_to_select_from_ranked.iloc[ranks_to_select_copy,:].index
            for rnk in ranks_to_select_copy:
                start_index=rnk
                steps=0
                steps_limits=50
                tmp_rnk_dict={}
                while len(tmp_rnk_dict)==0:
                    #if start_index > genes_to_select_from_ranked.shape[0]-1: break
                    temp_gns_to_check=list(genes_to_select_from_set.difference(checked_genes))
                    if len(temp_gns_to_check)==0:break
                    if steps > steps_limits : break
                    back_tracker=rnk-steps
                    temp_gn=None
                    try:
                        temp_gn=genes_to_select_from_ranked.index[start_index] if genes_to_select_from_ranked.index[start_index]  else genes_to_select_from_ranked.index[back_tracker] 
                    except IndexError as e:
                        temp_gn=None
                    if temp_gn is None : start_index+=1;steps+=1;continue
                    checked_genes.append(temp_gn)
                    target_mean = np.float(mean_series.loc[temp_gn])*fc
                    interval=0.1*target_mean
                    mod_mean_interval = [target_mean - interval, target_mean+ interval]
                    temp_matching_genes_list = gene_parameters_df[(gene_parameters_df.mean_expr >= mod_mean_interval[0]) &(gene_parameters_df.mean_expr <= mod_mean_interval[1])].index.tolist()
                    if not temp_matching_genes_list:start_index+=1;steps+=1; continue
                    matching_genes_set=all_genes_set.intersection(temp_matching_genes_list)
                    if not matching_genes_set : start_index+=1; steps+=1;continue
                    tmp_matching_gene=np.random.choice(list(matching_genes_set),1)[0]
                    tmp_rnk_dict[temp_gn]=tmp_matching_gene
                    ranks.append(start_index)
                if len(tmp_rnk_dict)==0: continue
                for tmp_k,tmp_v in tmp_rnk_dict.items():
                    genes.append(tmp_k)
                    matching_genes.append(tmp_v)
                    matching_genes_dict[tmp_k]=tmp_v
                    
                 
            
        #ranks_to_select=np.random.choice([rnk for rnk in range(len_genes_to_select_from_temp_list)],no_genes)
        del matching_genes_dict
        del genes_to_select_from_set
        del mean_series
        del all_genes_set
        del gene_parameters_df
        del profiled_fc   
        del checked_genes 
        del matched_genes_tracker
        inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,matched_cells])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
        out_dict['ranks'] = ranks
    else:
        
        gene_modification_profile_df=gene_modification_profile_df.astype({'genes':str})
        
        temp_gene_modification_profile_df = gene_modification_profile_df[(gene_modification_profile_df.fc == fc)
                                                                         & (gene_modification_profile_df.genes!='nan')]
        
        genes_to_select_from_set = genes_to_select_from_set.intersection(temp_gene_modification_profile_df.gene.tolist())
        
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        
        len_genes_to_select_from_temp_list=len(genes_to_select_from_temp_list)
        
        len_limit=no_genes if no_genes<=len_genes_to_select_from_temp_list else len_genes_to_select_from_temp_list
        
        checked_genes=[]
        
        matched_genes_tracker=set()
        
        while len(matching_genes_dict) < len_limit:
            
            temp_gns_to_check=list(genes_to_select_from_set.difference(checked_genes))
            
            if len(temp_gns_to_check)==0:break
            
            temp_gn=np.random.choice(temp_gns_to_check, 1)[0]
            
            checked_genes.append(temp_gn)
            
            matching_genes_list = [temp_gene_modification_profile_df[temp_gene_modification_profile_df.gene == temp_gn].genes.values[0]][0].split(',')
            
            if not matching_genes_list : continue
            
            matching_genes_set =all_genes_set.intersection(matching_genes_list)
            
            if not matching_genes_set : continue
            #np.random.seed(seed=1111)
            
            matching_genes_set_to_select=matching_genes_set.difference(matched_genes_tracker)
            
            matching_genes_set_to_select=matching_genes_set_to_select if len(matching_genes_set_to_select)>=1 else matching_genes_set
            
            matching_gene = np.random.choice(list(matching_genes_set_to_select), 1)[0]
            
            genes.append(temp_gn)
            
            matching_genes.append(matching_gene)
            
            matching_genes_dict[temp_gn] = matching_gene
            
            matched_genes_tracker.add(matching_gene)
            
        del matching_genes_list
        
        del temp_gene_modification_profile_df
            
        del genes_to_select_from_set
            
        del matching_genes_dict
            
        del genes_to_select_from_temp_list
            
        del checked_genes
        
        del matched_genes_tracker
        inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,np.random.choice(all_cells,len_cells,replace=False)])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
    return out_dict


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

def log_transform_dataframe(inputdata,base=2.718281828459045,pseudo_value=1):
    
    '''Log-transforms data frame using the provided base'''

    in_array=np.asarray(inputdata)

    logged_array=log_transform_array(input_array=in_array,base=base,pseudo_value=pseudo_value)

    outputdata=pd.DataFrame(logged_array,index=inputdata.index,columns=inputdata.columns)

    return outputdata

def log_transform_array(input_array,base=2.718281828459045,pseudo_value=1):
    
    '''Log-transforms array using the provided base'''

    logged_array=np.log(input_array+pseudo_value)/np.log(base)

    return logged_array


def plot_heatmap(data_frame,title='Heatmap',xlab='X-axis',ylab='Y-axis',labs_font_size=10,
                 xlabs_ratotation=70,ylabs_ratotation=0,cmap='YlGnBu'):

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


def get_list_of_specific_file_type(directory, file_pattern):
    '''Returns a list of certain file types given a directory and a pattern matching the file e.g its extension'''
    all_files = get_all_files_in_directory(directory);
    
    pattern=re.compile(file_pattern)

    specific_file_list = [];

    for current_file in all_files:

        if pattern.search(string=current_file)==None : continue

        specific_file_list.append(current_file)

    return specific_file_list

