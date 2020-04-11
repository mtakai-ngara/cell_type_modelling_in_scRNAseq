import numpy as np, gc,re,itertools,os
import hdbscan,umap
from sklearn import  metrics,neighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale 
import pandas as pd
from scanpy.api.pp import filter_genes_dispersion

#External R scripts
Rcmd='/usr/local/bin/Rscript'
var_genes_script='/mnt/kauffman/nbserver/notebooks/users/mtakai/R/test.R'
deseq_norm_script='/mnt/kauffman/nbserver/notebooks/users/mtakai/R/get_deseq_normalization.R'


## Step 1. Functions for reading,transformation and perturbing gene expression dataframe
def read_inputdata(filepath,sep='\t',header=0,index_col=0,nrows=None):
    '''Reads data table into a data frame'''
    input_df=pd.read_csv(filepath,sep=sep,header=header,index_col=index_col,nrows=nrows)
    return (input_df)

#prepare_data_log=open('prepare_data.log','w')
#@profile(stream=prepare_data_log)
def prepare_data(inputdata, fc, no_genes,  no_cells, method='multiplicative',quantiles=None,
                 filter_genes_pattern=None,genes_and_cells_cut_off=None,normalize=None,
                 pseudo_value=1,find_var_genes=None,preselected_genes=None,
                 gene_modification_profile_df=None,genes_in_all_cells=False):
    data = remove_nonexpressedgenes(inputdata)
    del inputdata
    gc.collect()
    if filter_genes_pattern:
        #Useful in cases where user may want to exclude e.g ERCC
        data = remove_genes(inputdata=data, gene_pattern=filter_genes_pattern)
    if normalize:
        #In case you provide counts instead of normalized expression
        norm_data_dict=get_genes_sf(counts_df=data,pseudo_value=pseudo_value,loc_func=np.median,
                                          return_norm_expr=True)
        data =norm_data_dict['norm']
        del norm_data_dict
        gc.collect()
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
                                 gene_modification_profile_df=gene_modification_profile_df)
    return perturbed_data

#perturb_data_fh=open('perturb_data.log','w')
#@profile(stream=perturb_data_fh)
def perturb_data(inputdata,fc,no_genes,no_cells,method='multiplicative',preselected_genes=None,
                 genes_in_all_cells=False,gene_modification_profile_df=None):
    '''Modifys expression data frame of a given set of genes, in target cells using the input method''' 
    all_cells=inputdata.columns
    #np.random.seed(seed=115678)
    cells=np.random.choice(all_cells,no_cells,replace=False)
    all_genes_set=set(inputdata.index)
    #np.random.seed(seed=None)
    genes_to_select_from_set=all_genes_set
    if preselected_genes:
        genes_to_select_from_set=genes_to_select_from_set.intersection(preselected_genes)
    if genes_in_all_cells == True:
        temp_target_cell_df=inputdata.loc[:,cells]
        temp_target_cells_genes_list=temp_target_cell_df.index[temp_target_cell_df[temp_target_cell_df>0.0].count(axis=1)
                                                            ==no_cells]
        genes_to_select_from_set=genes_to_select_from_set.intersection(temp_target_cells_genes_list)
        del temp_target_cell_df,temp_target_cells_genes_list
    del all_cells,preselected_genes
    gc.collect()
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
                                              all_genes_set=all_genes_set,genes_to_select_from_set=genes_to_select_from_set,
                                              gene_modification_profile_df=gene_modification_profile_df)
            out_dict['cells'] = cells
            out_dict['fc'] = fc
            out_dict['data'] = mod_dict['data']
            out_dict['genes'] = mod_dict['genes']
            out_dict['matching_genes'] = mod_dict['matching_genes']
            del mod_dict,all_genes_set
            gc.collect()
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
        del inputdata,fc,cells
        gc.collect()
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

def multiplicative_modification(inputdata,genes,cells,fc=2,mean_interval=2.0,drop_out_interval=0.1,cv2=None):
    '''Given specific samples and genes, modifys the expression levels based on the provided fold change'''
    inputdata.loc[genes, cells]= inputdata.loc[genes, cells].mul(fc)
    return inputdata

#modification_in_subspace_fh=open('modification_in_subspace.log','w')
#@profile(stream=modification_in_subspace_fh)
def modification_in_subspace(inputdata,cells,no_genes,fc,all_genes_set=None,genes_to_select_from_set=None,
                            gene_modification_profile_df=None):
    '''Given specific cells and number of genes, modifys the expression levels based on the provided fold change and
    ensures the mean is within a target subspace'''
    all_genes_set=set(inputdata.index) if all_genes_set is None else all_genes_set.intersection(inputdata.index)
    profiled_fc=None if gene_modification_profile_df is None else gene_modification_profile_df.fc.unique().tolist()
    genes_to_select_from_set=all_genes_set if genes_to_select_from_set is None else genes_to_select_from_set.intersection(all_genes_set)
    all_genes_set_as_list=list(all_genes_set)
    len_cells=len(cells)
    all_cells=inputdata.columns     
    genes=[]      
    matching_genes=[]
    matching_genes_dict={}
    out_dict={}
    matched_cells=np.random.choice(all_cells,len_cells,replace=False)
    if fc==0.0:
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        selected_genes=np.random.choice(genes_to_select_from_temp_list,no_genes,replace=False)
        out_dict['genes'] = selected_genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=selected_genes
        del genes_to_select_from_temp_list
        return out_dict
    elif profiled_fc is None or fc not in profiled_fc:
        gene_parameters_df=gene_parameters(inputdata.loc[all_genes_set,:])
        mean_series = gene_parameters_df.mean_expr
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        len_genes_to_select_from_temp_list=len(genes_to_select_from_temp_list)
        len_limit=no_genes if no_genes<=len_genes_to_select_from_temp_list else len_genes_to_select_from_temp_list
        checked_genes = []
        matched_genes_tracker=set()
        while len(matching_genes_dict)<=len_limit:
            temp_gns_to_check=list(genes_to_select_from_set.difference(checked_genes))
            if len(temp_gns_to_check)==0:break
            temp_gn=np.random.choice(temp_gns_to_check, 1)[0]
            checked_genes.append(temp_gn)
            target_mean = mean_series.loc[temp_gn]*fc
            interval=0.1*target_mean
            mod_mean_interval = [target_mean - interval, target_mean+ interval]
            temp_matching_genes_list = gene_parameters_df[(gene_parameters_df.mean_expr >= mod_mean_interval[0]) &
                                                          (gene_parameters_df.mean_expr <= mod_mean_interval[1])].index.tolist()
            if not temp_matching_genes_list: continue
            #np.random.seed(seed=1111)
            matching_genes_set=all_genes_set.intersection(temp_matching_genes_list)
            if not matching_genes_set : continue
            diff_matching_genes_set=matching_genes_set.difference(matched_genes_tracker)
            tmp_matching_gene=np.random.choice(list(diff_matching_genes_set),1)[0] if diff_matching_genes_set  else np.random.choice(list(matching_genes_set),1)[0]
            genes.append(temp_gn)
            matching_genes.append(tmp_matching_gene)
            matching_genes_dict[temp_gn]=tmp_matching_gene
            matched_genes_tracker.add(tmp_matching_gene)
        inputdata.loc[genes,cells]=inputdata.loc[matching_genes,matched_cells].values
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
        del matching_genes_dict,genes_to_select_from_set,mean_series,all_genes_set,gene_parameters_df
        del profiled_fc,checked_genes,matched_genes_tracker ,matched_cells,genes_to_select_from_temp_list
        del temp_matching_genes_list
        return out_dict
    else:
        gene_parameters_df=gene_parameters(inputdata.loc[all_genes_set,:])
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
        del matching_genes_list,temp_gene_modification_profile_df,genes_to_select_from_set    
        del matching_genes_dict,genes_to_select_from_temp_list,checked_genes,matched_genes_tracker
        #gc.collect()
        inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,np.random.choice(all_cells,len_cells,replace=False)])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
        del all_cells
        return out_dict
    pass

def modification_in_subspace_for_selected_genes(inputdata,cells,target_genes,fc,gene_parameters_df,all_genes_set=None,
                                                genes_to_select_from_set=None,gene_modification_profile_df=None):
    '''Given specific cells and genes, modifys the expression levels based on the provided fold change and
    ensures the mean is within a target subspace '''
    all_genes_set=set(inputdata.index.tolist()) if all_genes_set is None else all_genes_set.intersection(inputdata.index.tolist())
    genes_to_select_from_set=all_genes_set if genes_to_select_from_set is None else genes_to_select_from_set.intersection(all_genes_set)
    gene_parameters_df=gene_parameters_df.loc[all_genes_set,:]
    len_cells=len(cells)
    all_cells=inputdata.columns.tolist()     
    genes=[] 
    no_genes=len(target_genes)
    matching_genes=[]
    matching_genes_dict={}
    out_dict={}
    profiled_fc=None if gene_modification_profile_df is None else gene_modification_profile_df.fc.unique().tolist()
    out_dict={}
    if fc==0.0:
        out_dict['genes'] = target_genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=target_genes  
    elif profiled_fc is None or fc not in profiled_fc:
        mean_series = gene_parameters_df.mean_expr
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        len_genes_to_select_from_temp_list=len(genes_to_select_from_temp_list)
        len_limit=no_genes if no_genes<=len_genes_to_select_from_temp_list else len_genes_to_select_from_temp_list
        matched_genes_tracker=set()
        for temp_gn in target_genes:
            target_mean = np.float(mean_series.loc[temp_gn])*fc
            interval=0.1*target_mean 
            mod_mean_interval = [target_mean - interval, target_mean+ interval]
            temp_matching_genes_list = gene_parameters_df[(gene_parameters_df.mean_expr >= mod_mean_interval[0]) &
                                                          (gene_parameters_df.mean_expr <= mod_mean_interval[1])].index.tolist()
            if not temp_matching_genes_list: continue
            matching_genes_list=list(all_genes_set.intersection(temp_matching_genes_list))
            if not matching_genes_list : continue
            matching_genes_set=set(matching_genes_list).difference(matched_genes_tracker)
            matching_gene=np.random.choice(list(matching_genes_set),1)[0] if len(matching_genes_set)>=1 else np.random.choice(matching_genes_list,1)[0]
            #if not matching_gene: continue
            genes.append(temp_gn)
            matching_genes.append(matching_gene)
            matching_genes_dict[temp_gn]=matching_gene
            matched_genes_tracker.add(matching_gene)
        del matching_genes_dict
        del genes_to_select_from_set
        del mean_series  
        del all_genes_set 
        del gene_parameters_df  
        del profiled_fc   
        del matched_genes_tracker
        inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,
                                                          np.random.choice(all_cells,len_cells,replace=False)])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
                         
    else:
        gene_modification_profile_df=gene_modification_profile_df.astype({'genes':str})
        temp_gene_modification_profile_df = gene_modification_profile_df[(gene_modification_profile_df.fc == fc)
                                                                         & (gene_modification_profile_df.genes!='nan')]
        genes_to_select_from_set = genes_to_select_from_set.intersection(temp_gene_modification_profile_df.gene.tolist())
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        len_genes_to_select_from_temp_list=len(genes_to_select_from_temp_list)
        len_limit=no_genes if no_genes<=len_genes_to_select_from_temp_list else len_genes_to_select_from_temp_list
        matched_genes_tracker=set()
        for temp_gn in target_genes:
            matching_genes_list = [temp_gene_modification_profile_df[temp_gene_modification_profile_df.gene == temp_gn].genes.values[0]][0].split(',')
            if not matching_genes_list : continue
            matching_genes_set =all_genes_set.intersection(matching_genes_list)
            if not matching_genes_set : continue
            print( matching_genes_set)
            matching_genes_set_to_select=matching_genes_set.difference(matched_genes_tracker)
            matching_genes_set_to_select=matching_genes_set_to_select if len(matching_genes_set_to_select)>=1 else matching_genes_set
            matching_gene = np.random.choice(list(matching_genes_set_to_select), 1)[0]
            genes.append(temp_gn)
            matching_genes.append(matching_gene)
            matched_genes_tracker.add(matching_gene)
        
        
        del matching_genes_list
        del temp_gene_modification_profile_df    
        del genes_to_select_from_set 
        del matching_genes_dict 
        del genes_to_select_from_temp_list
        del matched_genes_tracker
        inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,
                                                          np.random.choice(all_cells,len_cells,replace=False)])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
        print('To confirm if its correct')
      
    return out_dict

def add_marker_gene_modification(inputdata,cells,no_genes,fc,gene_parameters_df,all_genes_set=None,
                                 genes_to_select_from_set=None,gene_modification_profile_df=None):
    
    '''Given specific cells and number of genes, modifys the expression levels based on the provided fold change and
    ensures the mean is within a target subspace and concatenates the markers to the expression table'''
    
    all_genes_set=set(inputdata.index.tolist()) if all_genes_set is None else all_genes_set.intersection(inputdata.index.tolist())
    profiled_fc=None if gene_modification_profile_df is None else gene_modification_profile_df.fc.unique().tolist()
    genes_to_select_from_set=all_genes_set if genes_to_select_from_set is None else genes_to_select_from_set.intersection(all_genes_set)
    gene_parameters_df=gene_parameters_df.loc[all_genes_set,:]
    len_cells=len(cells)
    all_cells=inputdata.columns.tolist() 
    non_selected_cells=set(all_cells).difference(cells)
    genes=[]      
    matching_genes=[]
    matching_genes_dict={}
    out_dict={}
    if fc==0.0:
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        selected_genes=np.random.choice(genes_to_select_from_temp_list,no_genes,replace=False).tolist()
        out_dict['genes'] = selected_genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=selected_genes 
    elif profiled_fc is None or fc not in profiled_fc:
        mean_series = gene_parameters_df.mean_expr
        genes_to_select_from_temp_list=list(genes_to_select_from_set)
        len_genes_to_select_from_temp_list=len(genes_to_select_from_temp_list)
        len_limit=no_genes if no_genes<=len_genes_to_select_from_temp_list else len_genes_to_select_from_temp_list
        checked_genes = []
        matched_genes_tracker=set()
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
            #np.random.seed(seed=1111)
            matching_genes_list=list(all_genes_set.intersection(temp_matching_genes_list))
            if not matching_genes_list : continue
            matching_genes_set=set(matching_genes_list).difference(matched_genes_tracker)
            matching_gene=np.random.choice(list(matching_genes_set),1)[0] if len(matching_genes_set)>=1 else np.random.choice(matching_genes_list,1)[0]
            #if not matching_gene: continue
            genes.append(temp_gn)
            matching_genes.append(matching_gene)
            matching_genes_dict[temp_gn]=matching_gene
            matched_genes_tracker.add(matching_gene)
        del matching_genes_dict
        del genes_to_select_from_set
        del mean_series
        del all_genes_set
        del gene_parameters_df
        del profiled_fc
        del checked_genes 
        del matched_genes_tracker
        #inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,np.random.choice(all_cells,len_cells,replace=False)])
        first_sp_array=np.array(inputdata.loc[genes,non_selected_cells])
        sec_sp_array=np.array(inputdata.loc[matching_genes,np.random.choice(all_cells,len_cells,replace=False)])
        first_and_sec_sp_array=np.concatenate([first_sp_array,sec_sp_array],axis=1)
        md_expr_cells=list(non_selected_cells)
        md_expr_cells.extend(cells)
        temp_mod_df=pd.DataFrame(first_and_sec_sp_array,index=['syn_marker_'+ str(md_gn_ind+1) for 
                                                               md_gn_ind in range(no_genes) ],
                                 columns=md_expr_cells)
        inputdata=pd.concat([inputdata,temp_mod_df.loc[:,inputdata.columns]])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
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
        #inputdata.loc[genes,cells]=np.array(inputdata.loc[matching_genes,np.random.choice(all_cells,len_cells,replace=False)])
        first_sp_array=np.array(inputdata.loc[genes,non_selected_cells])
        sec_sp_array=np.array(inputdata.loc[matching_genes,np.random.choice(all_cells,len_cells,replace=False)])
        first_and_sec_sp_array=np.concatenate([first_sp_array,sec_sp_array],axis=1)
        md_expr_cells=list(non_selected_cells)
        md_expr_cells.extend(cells)
        temp_mod_df=pd.DataFrame(first_and_sec_sp_array,index=['syn_marker_'+ str(md_gn_ind+1) for 
                                                               md_gn_ind in range(no_genes) ],
                                 columns=md_expr_cells)
        inputdata=pd.concat([inputdata,temp_mod_df.loc[:,inputdata.columns]])
        out_dict['genes'] = genes
        out_dict['cells'] = cells
        out_dict['fc'] = fc
        out_dict['data'] = inputdata
        out_dict['matching_genes']=matching_genes
    return out_dict
    

def mean_modification(inputdata,genes,cells,fc=2):
    ''' Method modifys the mean expression of target genes and cells based on a fold change  
    and generates normally distributed expression '''
    inputdata_mean_series=inputdata.mean(axis=1)
    inputdata.loc[genes, cells] = [np.random.normal(loc=inputdata_mean_series.loc[gn]*fc,
                                           size=len(cells))for gn in genes]
    return inputdata

def add_synthetic_genes_modification(inputdata,genes,cells,fc=2,std=1.0):
    
    '''Given specific samples and genes, modifys the expression levels based on their mean expression 
    by adding synthetic genes provided fold change.'''
    inputdata_mean_series=inputdata.mean(axis=1)
    all_scs=inputdata.columns
    target_genes_df = pd.DataFrame([[random.gauss(mu=inputdata_mean_series.loc[gn]*fc, sigma=std) 
                                     if sc in cells else random.gauss(mu=inputdata_mean_series.loc[gn], sigma=std)
                                     for sc in all_scs] for gn in genes],index=['synthetic_gene_'+str(ind+1) 
                                                                                             for ind in range(len(genes))],
                                  columns=inputdata.columns)
    outputdata=inputdata.append(target_genes_df)
    return outputdata


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
    if return_norm_expr is True:
        norm_expr_df=counts_df.div(samples_sf,axis=1)
        out_dict['norm'] = norm_expr_df
    return out_dict

def deseq_norm_subprocess(inputfile,output_dir):
    '''Uses the R implementation of the DESeq normalization'''
    cmd=[Rcmd,deseq_norm_script, '-i', inputfile, '-o', output_dir]
    #norm_genes_fname=subprocess.check_output(cmd,shell=False,universal_newlines=True).split(' ')[1]
    norm_genes_fname=subprocess.check_output(cmd,shell=False,universal_newlines=True).split(' ')[1]
    #norm_genes_fname=norm_genes_fname.strip('"').strip('"').strip('\n')
    #norm_genes_fname=norm_genes_fname.strip('"')
    return norm_genes_fname

def genes_quantiles(inputdata,lower_quantile=0.0,upper_quantile=1.0):
    '''returns the input data frame after selecting gene from the specific mean-quantiles '''
    mean_series=inputdata.mean(axis=1)
    lower_cut_off=mean_series.quantile(np.float(lower_quantile))
    upper_cut_off = mean_series.quantile(np.float(upper_quantile))
    mean_series_filt=mean_series[mean_series>=lower_cut_off]
    if np.round(upper_quantile,2)<1.00:
        mean_series_filt=mean_series_filt[mean_series_filt < upper_cut_off]
        output_data=inputdata.loc[mean_series_filt.index,inputdata.columns]
        return  output_data  
    else:
        mean_series_filt[mean_series_filt <= upper_cut_off]
        output_data=inputdata.loc[mean_series_filt.index,inputdata.columns]
        return output_data
    pass

def filter_genes_and_samples(inputdata,cut_off=0,no_samples=1,genes_count_cut_off=1):
    '''Filters genes based on the cut off sample count(s) from a dataframe'''
    cut_off=float(cut_off)
    no_samples=int(no_samples)
    genes_count_cut_off=int(genes_count_cut_off)
    selected_samples=inputdata.columns[inputdata[inputdata>cut_off].count(axis=0)>=genes_count_cut_off]
    local_no_samples=no_samples if len(selected_samples)>=no_samples else len(selected_samples)
    selected_gns=inputdata.index[inputdata[inputdata>cut_off].count(axis=1)>=local_no_samples]
    outputdata=inputdata.loc[selected_gns,selected_samples]
    return outputdata

def remove_nonexpressedgenes(inputdata): 
    '''Filters genes with zero expression in all samples'''
    selected_gns=inputdata.index[inputdata[inputdata>0.0].count(axis=1)>=1]
    selected_samples=inputdata.columns[inputdata[inputdata>0.0].count(axis=0)>=1]
    output_data=inputdata.loc[selected_gns,selected_samples]
    return output_data

def remove_genes(inputdata, gene_pattern):
    '''Removes genes whose ids or names mateches the provided pattern. Good for removing e.g ERCCs'''
    pattern=re.compile(gene_pattern)
    genes_to_keep=[gn for gn in inputdata.index if pattern.search(string=gn) is None]
    outdata=inputdata.loc[genes_to_keep,:]
    return outdata

# Step 3. Computational strategies to cluster and evaluate clusters
def run_kmeans(input_array,no_clusters=2,scale_array=True):
    ''' Runs K-means clustering on the input array'''
    #np.random.seed(2573780)
    data = scale(input_array) if scale_array else input_array
    kmean = KMeans(init='k-means++', n_clusters=no_clusters, n_init=10).fit(data)
    return kmean

def run_knn(input_array,labels,no_neighbours=5, radius=1.0,algorithm='auto',leaf_size=30,distance_metric='minkowski',
            p=2,cpus=1):
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

def run_pca(input_array,components=None):  
    '''Runs PCA on input array'''
    #np.random.seed(2573780)
    in_array= np.array(input_array)
    del input_array
    gc.collect()
    in_array_dim=in_array.shape
    final_component = in_array_dim[0] if components is None or components > in_array_dim[0] else components 
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
        empty_array = np.empty((in_array_dim[0], final_component))
        empty_array[:] = np.NAN        
        pca_res_dict['pca_array']=empty_array        
        return pca_res_dict
    else:       
        return pca_res_dict
    pass

def run_pca_on_df(input_df,components=None,transpose=False):
    '''Runs PCA reduction on multi-dimensional data frame'''
    #np.random.seed(2573780)
    input_array= input_df.values
    final_input_array= input_array if not transpose else np.transpose(input_array)
    samples_names=input_df.columns.tolist() if transpose else input_df.index.tolist()
    del input_array
    del input_df
    final_input_array_dim=final_input_array.shape
    final_component = final_input_array_dim[0] if components == None or components > final_input_array_dim[0] else components
    #pca_res_dict=run_pca(input_array=final_input_array,components=final_component,method='incremental_pca')
    pca_res_dict=run_pca(input_array=final_input_array,components=final_component)
    pca_array=pca_res_dict['pca_array']
    if not np.isnan(pca_array).all():
        pca_df=pd.DataFrame(pca_array,index=samples_names,
                            columns=['PC'+str(m+1) for m in range(final_component)])
        pca_res_dict['pca_df'] = pca_df
    return pca_res_dict 

def run_pca_test(input_array,components=None,method='pca'):
    '''Runs PCA on input array'''
    #np.random.seed(2573780)
    in_array= np.array(input_array)
    in_array_dim=input_array.shape
    final_component = in_array_dim[0] if components is None or components > in_array_dim[0] else components
    pca_res_dict={}
    try:
        
        if method=='ipca':
            model = IncrementalPCA(n_components=final_component, batch_size=10)
            ipca_res = model.fit_transform(in_array)
            cumulative_variance = np.cumsum(np.round(ipca_res.explained_variance_ratio_, decimals=4) * 100)
            sklearn_tranfrom=sklearn_pca.fit_transform(in_array)
            pca_res_dict['cum_variance']=cumulative_variance
            pca_res_dict['pca_array']=ipca_res
            
        elif method=='pca':
            sklearn_pca = PCA(n_components=final_component)
            sklearn_pca.fit(in_array)
            cumulative_variance = np.cumsum(np.round(sklearn_pca.explained_variance_ratio_, decimals=4) * 100)
            sklearn_tranfrom=sklearn_pca.fit_transform(in_array)
            pca_res_dict['cum_variance']=cumulative_variance
            pca_res_dict['pca_array']=sklearn_pca
            
            
        else:
            sklearn_pca = PCA(n_components=final_component)
            sklearn_pca.fit(in_array)
            cumulative_variance = np.cumsum(np.round(sklearn_pca.explained_variance_ratio_, decimals=4) * 100)
            sklearn_tranfrom=sklearn_pca.fit_transform(in_array)
            pca_res_dict['cum_variance']=cumulative_variance
            pca_res_dict['pca_array']=sklearn_pca
            
       
    except :
        
        empty_array = np.empty((in_array_dim[0], final_component))
        empty_array[:] = np.NAN
        pca_res_dict['pca_array']=empty_array
        return pca_res_dict
    else:
        
        return pca_res_dict
    pass

def run_fast_pca_on_df(input_df,components=None):
    '''Runs PCA reduction on multi-dimensional data frame'''
    sklearn_pca = PCA(n_components=components)
    sklearn_tranfrom=sklearn_pca.fit_transform(input_df)
    return sklearn_tranfrom 

# Step 2. Functions for finding most variable genes
def variable_genes(counts_df,spike_ins_ids=None,pseudo_value=1,mean_quantile = .95,
                   cv_squared = 0.3,technical_fit_start=-2,technical_fit_end=6,technical_fit_size=1000,
                   multiple_test_correction='fdr_bh',p_adjust_cut_off=.1,method='brenneck'):
    '''Uses different methods to pick significantly variable genes'''
    counts_df=remove_nonexpressedgenes(inputdata=counts_df)
    if method=='brenneck':
        #Uses the Brennecke et al (2013) method
        var_genes_dict=vargenes.variable_genes_brenneck(counts_df=counts_df,spike_ins_ids=spike_ins_ids,
                                                        pseudo_value=pseudo_value,mean_quantile = mean_quantile,
                                                        cv_squared = cv_squared,
                                                        technical_fit_start=technical_fit_start,
                                                        technical_fit_end=technical_fit_end,
                                                        technical_fit_size=technical_fit_size,
                                                        multiple_test_correction=multiple_test_correction,
                                                        p_adjust_cut_off=p_adjust_cut_off)
        return var_genes_dict
    pass

def variable_genes_subprocess(inputfile,output_dir):
    '''Uses the R implementation of the Brennecke method as a subprocess to find variable genes'''
    cmd=[Rcmd,var_genes_script, '-i', inputfile, '-o', output_dir]
    genes_var_fname=subprocess.check_output(cmd,shell=False,universal_newlines=True).split(' ')[1]
    genes_var_fname=genes_var_fname.strip('"').strip('"').strip('\n')
    genes_var_fname=genes_var_fname.strip('"')
    return genes_var_fname

def variable_genes_test_subprocess(inputfile,norm_inputfile,output_dir,top_variable_genes=100,prefix='var_genes'):
    
    '''Uses the R implementation of the Brennecke method as a subprocess to find variable genes'''
    
    Rcmd='/usr/local/bin/Rscript'
    
    var_genes_script='/mnt/kauffman/nbserver/notebooks/users/mtakai/R/var_genes_with_norm.R'
   
    cmd=[Rcmd,var_genes_script, '-i', inputfile,'-n',norm_inputfile,'-o', output_dir,'-t',str(top_variable_genes),'-p',prefix]
    
    genes_var_fname=subprocess.check_output(cmd,shell=False,universal_newlines=True).split(' ')[1]
    
    genes_var_fname=genes_var_fname.strip('"').strip('"').strip('\n')
    
    genes_var_fname=genes_var_fname.strip('"')
    
    return genes_var_fname


def variable_genes_test_rpy_interface(counts_df, norm_df, output_dir, fdr=0.05, minBiolDisp=0.5,
                                   prefix='var_genes', write_output=False, winsorization=True, quant_cutoff=0.9,
                                   min_cv2=0.3, n_ercc_min=2, min_count=2, min_prop=0.5, batchCorrected=False,
                                   rm_low_mean=0, nTopGenes=100, spike_ins_pattern=None):
    
    '''Uses the R implementation of the Brennecke method as a subprocess to find variable genes'''
    functions_string = None
    with open('/mnt/kauffman/nbserver/notebooks/users/mtakai/R/variable_genes.R', 'r') as fh:
        functions_string = fh.read()
    var_genes_package = SignatureTranslatedAnonymousPackage(functions_string, "var_genes_package")
    
    brenneck_var_genes_funtion = var_genes_package.Brennecke_get_variableGenes
    spike_ins = None
    if spike_ins_pattern:
        ercc_pattern = re.compile(spike_ins_pattern)
        spike_ins = [sp for sp in counts_df.index.tolist() if not ercc_pattern.search(string=sp) is None]
    r_norm_df = pandas2ri.py2ri(norm_df)
    r_counts_df = pandas2ri.py2ri(counts_df)
    r_var_genes_mat=None
    ro.r['options'](warn=-1)
    try:
   
        if spike_ins is None or len(spike_ins)<10:
            r_var_genes_mat = brenneck_var_genes_funtion(count_table=r_counts_df, norm_table=r_norm_df,
                                                         prefix=prefix,fdr=fdr, minBiolDisp=minBiolDisp,
                                                         winsorization=winsorization,min_cv2=min_cv2,
                                                         n_ercc_min=n_ercc_min, min_count=min_count, min_prop=min_prop,
                                                         quant_cutoff=quant_cutoff, batchCorrected=batchCorrected,
                                                         outputDIR=output_dir,
                                                         nTopGenes=nTopGenes, rm_low_mean=rm_low_mean, 
                                                         write_output=write_output)
        else:
            r_spike_ins = ro.vectors.StrVector(spike_ins)
            r_var_genes_mat = brenneck_var_genes_funtion(count_table=r_counts_df, norm_table=r_norm_df, spikes=r_spike_ins,
                                                         prefix=prefix,fdr=fdr, minBiolDisp=minBiolDisp,
                                                         winsorization=winsorization,
                                                         min_cv2=min_cv2,n_ercc_min=n_ercc_min, min_count=min_count,
                                                         min_prop=min_prop,quant_cutoff=quant_cutoff, 
                                                         batchCorrected=batchCorrected,outputDIR=output_dir,
                                                         nTopGenes=nTopGenes, rm_low_mean=rm_low_mean,
                                                         write_output=write_output)
        p_var_genes_df = pandas2ri.ri2py(r_var_genes_mat)
        return p_var_genes_df
    except Exception as err:
        print(err, 'Error thrown when running the brenneck_var_genes_funtion')
        pass
    pass

#cluster_data_fh=open('cluster_data.log','w')
#@profile(stream=cluster_data_fh)
def cluster_data(inputdata,labels,transpose_inputdata=False,discovery_method='pca',pca_components=2,
                 clustering_method='knn',knn_neighbours=3,knn_radius=1.0,knn_algorithm='auto',
                 knn_leaf_size=30,knn_distance_metric='minkowski',knn_p=2,knn_cpus=1,
                 kmeans_clusters=2,scale_kmeans=True,tsne_components=2,tsne_perplexity=30,
                 umap_neighbors=5,umap_min_dist=0.1,umap_components=2,umap_metric='euclidean',
                 corr_method='spearman',dbscan_min_cluster_size=5,dbscan_min_samples=None,dbscan_metric='euclidean'):
    '''Runs the data through the specific clustering and computes the homogeneity score '''
    out_list=[]
    if discovery_method=='direct':
        try:
            hs_res_dict=get_homogeneity_score(input_array=inputdata.transpose().values,labels=labels,
                                              clustering_method=clustering_method,
                                              knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                              knn_leaf_size=knn_leaf_size,
                                              knn_distance_metric=knn_distance_metric,knn_p=knn_p,knn_cpus=knn_cpus,
                                              kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs'] 
            predictions_dict=dict(zip(inputdata.columns.tolist(),predictions_labs))   
            modification_dict=dict(zip(inputdata.columns.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in inputdata.columns.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['direct',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,
                             calinski_harabaz_score,cosine_silhouette_score,euclidean_silhouette_score,
                             tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del hs_res_dict
            gc.collect()
            return out_list
    
        except:
            out_list.append(['direct',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    
    if discovery_method=='pca':        
        try:
            pca_res=run_pca_on_df(input_df=inputdata,components=pca_components,transpose=True)            
            pca_df=pca_res['pca_df']
            hs_res_dict=get_homogeneity_score(input_array=pca_df.values,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs'] 
            predictions_dict=dict(zip(pca_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(pca_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in pca_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['pca',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,calinski_harabaz_score,
                             cosine_silhouette_score,euclidean_silhouette_score,tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del pca_res,pca_df
            gc.collect()
            return out_list
        except:
            out_list.append(['pca',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
            
    if discovery_method=='pca_tsne':
        try:
            pca_res=run_pca_on_df(input_df=inputdata,components=pca_components,transpose=True)
            pca_df=pca_res['pca_df']
            tsne_res = TSNE(n_components=tsne_components,verbose=0,perplexity=tsne_perplexity).fit_transform(pca_df)
            hs_res_dict=get_homogeneity_score(input_array=tsne_res,labels=labels,clustering_method=clustering_method,
                                              knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                              knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,knn_p=knn_p,
                                              knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs']
            predictions_dict=dict(zip(pca_df.index,predictions_labs))   
            modification_dict=dict(zip(pca_df.index,labels)) 
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in pca_df.index:    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['pca_tsne',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,
                             fm_score,calinski_harabaz_score,cosine_silhouette_score,euclidean_silhouette_score,
                             tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del hs_res_dict,inputdata,predictions_dict,modification_dict
            gc.collect()
            return out_list
            
        except:
            out_list.append(['pca_tsne',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
        
    if discovery_method=='umap':
        try:
            umap_array = umap.UMAP(n_neighbors=umap_neighbors,min_dist=umap_min_dist,n_components=umap_components,
                                   metric=umap_metric).fit_transform(inputdata.transpose())
            hs_res_dict=get_homogeneity_score(input_array=umap_array,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs'] 
            predictions_dict=dict(zip(inputdata.columns.tolist(),predictions_labs))   
            modification_dict=dict(zip(inputdata.columns.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in inputdata.columns.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append([discovery_method,hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,
                             calinski_harabaz_score,
                             cosine_silhouette_score,euclidean_silhouette_score,tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del inputdata,umap_array
            gc.collect()
            return out_list
        except:
            out_list.append([discovery_method,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    if discovery_method=='pca_umap':
        try:
            pca_res=run_pca_on_df(input_df=inputdata,components=pca_components,transpose=True)
            pca_df=pca_res['pca_df']
            umap_array = umap.UMAP(n_neighbors=umap_neighbors,min_dist=umap_min_dist,n_components=umap_components,
                                   metric=umap_metric).fit_transform(pca_df)
            hs_res_dict=get_homogeneity_score(input_array=umap_array,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs'] 
            predictions_dict=dict(zip(pca_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(pca_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in pca_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append([discovery_method,hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,
                             calinski_harabaz_score,
                             cosine_silhouette_score,euclidean_silhouette_score,tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del inputdata,umap_array,pca_res
            gc.collect()
            return out_list
        except:
            out_list.append([discovery_method,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    if discovery_method=='pca_max':
        
        try:
            pca_res=run_pca_on_df(input_df=inputdata,components=pca_components,transpose=True)
            pca_df=pca_res['pca_df']
            pca_max_dict=pc_pairs_homogeneity_scores_knn(pca_df=pca_df,labels=labels,neighbours=knn_neighbours,
                                                         radius=knn_radius,algorithm=knn_algorithm,leaf_size=knn_leaf_size,
                                                         distance_metric=knn_distance_metric,p=knn_p,cpus=knn_cpus)
            
            hs_dict=pca_max_dict['pc_scores']
            labs_dict=pca_max_dict['labs']
            predictions_labs=None
            hs=max(hs_dict.values())
            for it in hs_dict.items():
                if it[1]==hs:
                    predictions_labs=labs_dict[it[0]]
                    break
            
            predictions_dict=dict(zip(pca_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(pca_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in pca_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)         
            out_list.append(['pca_max',hs,tpr,fpr,fdr,acc,tn, fp, fn, tp])
            
            del pca_res,hs_dict,hs
            gc.collect()
            return out_list
                 
        except:
            out_list.append(['pca_max',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
            
    if discovery_method=='tsne':
        try:
            direct_tsne_res = TSNE(n_components=tsne_components,verbose=0,
                                   perplexity=tsne_perplexity).fit_transform(inputdata.transpose())
            direct_tsne_array=np.array(direct_tsne_res)
            hs_res_dict=get_homogeneity_score(input_array=direct_tsne_array,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs']
            predictions_dict=dict(zip(inputdata.columns.tolist(),predictions_labs))   
            modification_dict=dict(zip(inputdata.columns.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in inputdata.columns.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            
            out_list.append(['tsne',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,calinski_harabaz_score,
                             cosine_silhouette_score,euclidean_silhouette_score,tpr,fpr,fdr,acc,tn, fp, fn, tp])
            
            del direct_tsne_res,direct_tsne_array,hs,hs_res_dict
            gc.collect()
            return out_list
        except: 
            out_list.append(['tsne',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
        
    
    if discovery_method=='direct_corr':
        try:
            direct_corr_df=inputdata.corr(method=corr_method)
            hs_res_dict=get_homogeneity_score(input_array=direct_corr_df.values,labels=labels,
                                              clustering_method=clustering_method,knn_neighbours=knn_neighbours,
                                              knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                              knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,knn_p=knn_p,
                                              knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs']
            predictions_dict=dict(zip(direct_corr_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(direct_corr_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in direct_corr_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['direct_corr',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,
                             calinski_harabaz_score,cosine_silhouette_score,euclidean_silhouette_score,
                             tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del [direct_corr_df,hs,hs_res_dict]
            return out_list
        except:
            out_list.append(['direct_corr',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    if discovery_method=='pca_corr':
        try:
            pca_res=run_pca_on_df(input_df=inputdata,components=pca_components,transpose=True)
            pca_corr_df=pca_res['pca_df'].transpose().corr(method=corr_method)
            hs_res_dict=get_homogeneity_score(input_array=pca_corr_df,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs']
            predictions_dict=dict(zip(pca_corr_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(pca_corr_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in pca_corr_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['pca_corr',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,calinski_harabaz_score,
                             cosine_silhouette_score,euclidean_silhouette_score,tpr,fpr,fdr,acc,tn, fp, fn, tp])
            del [pca_res,hs,pca_corr_df,hs_res_dict]
            return out_list
        except:
            out_list.append(['pca_corr',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    if discovery_method=='corr_tsne':
        try:
            corr_tsne_df=inputdata.corr(method=corr_method)
            tsne_res = TSNE(n_components=tsne_components,verbose=0,perplexity=tsne_perplexity).fit_transform(corr_tsne_df)
            tsne_array=np.array(tsne_res)
            hs_res_dict=get_homogeneity_score(input_array=tsne_array,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs']
            predictions_dict=dict(zip(corr_tsne_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(corr_tsne_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in corr_tsne_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['corr_tsne',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,
                             calinski_harabaz_score,cosine_silhouette_score,euclidean_silhouette_score,tpr,
                             fpr,fdr,acc,tn, fp, fn, tp])
            
            del [corr_tsne_df,tsne_res,tsne_array,hs_res_dict]
            return out_list
        except:
            out_list.append(['corr_tsne',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    if discovery_method=='pca_corr_tsne':
        try:
            pca_res=run_pca_on_df(input_df=inputdata,components=pca_components,transpose=True)
            pca_corr_df=pca_res['pca_df'].transpose().corr(method=corr_method)
            tsne_res = TSNE(n_components=tsne_components,verbose=0,perplexity=tsne_perplexity).fit_transform(pca_corr_df.values)
            tsne_array=np.array(tsne_res)
            hs_res_dict=get_homogeneity_score(input_array=tsne_array,labels=labels,clustering_method=clustering_method,
                                     knn_neighbours=knn_neighbours,knn_radius=knn_radius,knn_algorithm=knn_algorithm,
                                     knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                                     knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans)
            hs=hs_res_dict['hs']
            complet_score=hs_res_dict['complet_score']
            v_score=hs_res_dict['v_score']
            adj_rand_index=hs_res_dict['adj_rand_index']
            adj_mutual_info=hs_res_dict['adj_mutual_info']
            fm_score=hs_res_dict['fm_score']
            calinski_harabaz_score=hs_res_dict['calinski_harabaz_score']
            cosine_silhouette_score=hs_res_dict['cosine_silhouette_score']
            euclidean_silhouette_score=hs_res_dict['euclidean_silhouette_score']
            predictions_labs=hs_res_dict['labs']
            predictions_dict=dict(zip(pca_corr_df.index.tolist(),predictions_labs))   
            modification_dict=dict(zip(pca_corr_df.index.tolist(),labels))
            [tn, fp, fn, tp]=[0,0,0,0]  
            for tmp_sc in pca_corr_df.index.tolist():    
                if modification_dict[tmp_sc]=='mod':
                    if predictions_dict[tmp_sc]=='mod':
                        tp+=1
                    else:
                        fn+=1
                else:
                    if predictions_dict[tmp_sc]=='unmod':
                        tn+=1
                    else:
                        fp+=1
            tpr=np.round(tp/(tp+fn),2)
            fpr=np.round(fp/(fp+tn),2)
            fnr=np.round(fn/(tp+fn),2)
            fdr=np.round(fp/(tp+fp),2)
            ppv=np.round(tp/(tp+fp),2)
            npv=np.round(tn/(tn+fn),2)
            acc=np.round((tp+tn)/(tp+fp+fn+tn),2)
            out_list.append(['pca_corr_tsne',hs,complet_score,v_score,adj_rand_index,adj_mutual_info,fm_score,
                             calinski_harabaz_score,cosine_silhouette_score,euclidean_silhouette_score,tpr,
                             fpr,fdr,acc,tn, fp, fn, tp])
            
            del [hs,pca_res,pca_corr_df,tsne_res,tsne_array]
            return out_list
            
        except: 
            out_list.append(['pca_corr_tsne',np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,
                             np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])
            return out_list
    del inputdata 
    gc.collect()
    pass

#pertub_data_get_var_genes_and_homogeneity_fh=open('pertub_data_get_var_genes_and_homogeneity.log','w')
#@profile(stream=pertub_data_get_var_genes_and_homogeneity_fh)
def pertub_data_get_var_genes_and_homogeneity(norm_indata,fc,no_genes,no_cells,modification_method='multiplicative',
                                              discovery_method='direct',quantiles=None,filter_genes_pattern=None,
                                              genes_and_cells_cut_off=None,normalize=None,pseudo_value=1,find_var_genes=None,
                                              pca_components=5,clustering_method='knn',knn_neighbours=3,knn_radius=1.0,
                                              knn_algorithm='auto',knn_leaf_size=30,knn_distance_metric='minkowski',knn_p=2,
                                              knn_cpus=1,kmeans_clusters=2,scale_kmeans=True,
                                              tsne_components=2,tsne_perplexity=30,umap_neighbors=5,umap_min_dist=0.1,
                                              umap_components=2,umap_metric='euclidean',corr_method='spearman',
                                              randomize_cells_labels=False,preselected_genes=None,preselect_quantile=None,
                                              nTopExpressed=None,gene_modification_profile_df=None,genes_in_all_cells=False,
                                              spike_ins_pattern='^gERCC-',nTopGenes=1500,dbscan_min_cluster_size=5,
                                              dbscan_min_samples=None,dbscan_metric='euclidean',
                                              post_mod_genes=None,run_id=None):
    '''Pertubs data and runs the specified clustering approach then gives an output of the homogeineity score plus\
    other meta info'''
    #if run_id:
        #print('Processing',run_id,'fc:',fc,'gns:',no_genes,'cells',no_cells,discovery_method,modification_method,
                    #'components',pca_components)
    try:
        input_dim=norm_indata.shape
        pertubed_data_dict=prepare_data(inputdata=norm_indata, fc=fc, no_genes=no_genes,no_cells=no_cells,
                                        method= modification_method,quantiles=quantiles,
                                        filter_genes_pattern=filter_genes_pattern,
                                        genes_and_cells_cut_off=genes_and_cells_cut_off,normalize=normalize,
                                        pseudo_value=pseudo_value,find_var_genes=find_var_genes,
                                        preselected_genes=preselected_genes,
                                        gene_modification_profile_df=gene_modification_profile_df,
                                        genes_in_all_cells=genes_in_all_cells)
        mod_gens = pertubed_data_dict['genes']
        perturbed_data_df=pertubed_data_dict['data']
        mod_cells =pertubed_data_dict['cells']
        matching_genes=pertubed_data_dict['matching_genes']
        del pertubed_data_dict
        gc.collect()
        var_genes_mat=filter_genes_dispersion(data=perturbed_data_df.transpose().values,flavor='seurat',min_disp=None,
                                              max_disp=None,min_mean=None, max_mean=None,n_bins=20,n_top_genes=nTopGenes,
                                              log=False,copy=False)
        var_genes_df=pd.DataFrame(var_genes_mat,index=perturbed_data_df.index)
        top_var_genes_df=var_genes_df[var_genes_df.gene_subset==True]
        top_var_genes_list=top_var_genes_df.index.tolist()
        mod_and_var_genes =set(mod_gens).intersection(top_var_genes_list)
        len_mod_and_var_genes=len(mod_and_var_genes) if not mod_and_var_genes is None else 0
        del var_genes_df,var_genes_mat,top_var_genes_df
        gc.collect()
        if post_mod_genes=='var_and_mod':
            top_genes_plus_all_mod_genes_list=top_var_genes_list
            top_genes_plus_all_mod_genes_list.extend(mod_gens)
            top_genes_plus_all_mod_genes_set=set(top_genes_plus_all_mod_genes_list)
            perturbed_data_df=perturbed_data_df.loc[top_genes_plus_all_mod_genes_set,:]
            del top_genes_plus_all_mod_genes_set,top_genes_plus_all_mod_genes_list
        elif post_mod_genes=='mod':
            perturbed_data_df=perturbed_data_df.loc[mod_gens,:]
        elif post_mod_genes=='sig_and_mod':
            sig_top_var_genes_set=set(top_var_genes_list)
            sig_top_var_genes_set.update(mod_gens)
            perturbed_data_df=perturbed_data_df.loc[sig_top_var_genes_set,:]
        elif post_mod_genes=='sig_only':
            perturbed_data_df=perturbed_data_df.loc[top_var_genes_list,:]
        elif post_mod_genes=='all':
            pass
        else:
            perturbed_data_df=perturbed_data_df.loc[top_var_genes_list,:]
        perturbed_data_df=pd.DataFrame(np.log1p(perturbed_data_df.values),
                                       index=perturbed_data_df.index,
                                       columns=perturbed_data_df.columns)
        del top_var_genes_list,mod_and_var_genes
        gc.collect()
        input_labs=['mod' if lb in mod_cells else 'unmod' for lb in perturbed_data_df.columns]  
        if randomize_cells_labels is True:
            np.random.shuffle(input_labs)
        temp_res=cluster_data(inputdata=perturbed_data_df,labels=input_labs,transpose_inputdata=True,
                              discovery_method=discovery_method,pca_components=pca_components,
                              clustering_method=clustering_method,knn_neighbours=knn_neighbours,knn_radius=knn_radius,
                              knn_algorithm=knn_algorithm,knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                              knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans,
                              tsne_components=tsne_components,tsne_perplexity=tsne_perplexity,umap_neighbors=umap_neighbors,
                              umap_min_dist=umap_min_dist,umap_components=umap_components,umap_metric=umap_metric,
                              corr_method=corr_method,dbscan_min_cluster_size=dbscan_min_cluster_size,
                              dbscan_min_samples=dbscan_min_samples,dbscan_metric=dbscan_metric)
        out_list=temp_res[0]
        del temp_res
        gc.collect()
        other_params=':'.join([str(par) for par in [quantiles,filter_genes_pattern,genes_and_cells_cut_off,normalize,
                                                    pseudo_value,find_var_genes,knn_cpus,
                                                    'genes_in_all_cells:'+str(genes_in_all_cells)] if par is not None])
        out_list.extend([no_cells, no_genes, fc, run_id, mod_cells, mod_gens,len_mod_and_var_genes,
                         modification_method, clustering_method,randomize_cells_labels,matching_genes, other_params,
                         pca_components,knn_neighbours,knn_radius,knn_algorithm,knn_leaf_size,knn_distance_metric,knn_p,
                         kmeans_clusters,scale_kmeans,tsne_components,tsne_perplexity,corr_method,nTopGenes,
                         ':'.join([str(d) for d in input_dim])])
        del len_mod_and_var_genes,mod_gens,other_params
        gc.collect()
        #semaphore.release()
        return out_list
    except KeyboardInterrupt:
        #semaphore.release()
        return 'KeyboardInterrupt'
    except Exception as err:
        #semaphore.release()
        print("Unexpected error during the pertubation and var gene detecton!!", err)
        return 
    pass

def pertub_data_get_var_genes_and_homogeneity_for_replicate_expt(norm_indata,fc,genes,cells,modification_method='multiplicative',
                                                                 discovery_method='direct',quantiles=None,
                                                                 filter_genes_pattern=None,genes_and_cells_cut_off=None,
                                                                 normalize=None,pseudo_value=1,find_var_genes=None,
                                                                 pca_components=5,clustering_method='knn',
                                                                 knn_neighbours=3,knn_radius=1.0,knn_algorithm='auto',
                                                                 knn_leaf_size=30,knn_distance_metric='minkowski',knn_p=2,
                                              knn_cpus=1,kmeans_clusters=2,scale_kmeans=True,
                                              tsne_components=2,tsne_perplexity=30,corr_method='spearman',
                                              randomize_cells_labels=False,preselected_genes=None,preselect_quantile=None,
                                              nTopExpressed=None,gene_modification_profile_df=None,
                                              gene_parameters_df=None,genes_in_all_cells=False,
                                              spike_ins_pattern='^gERCC-',nTopGenes=1500,dbscan_min_cluster_size=5,
                                              dbscan_min_samples=None,dbscan_metric='euclidean',
                                              post_mod_genes=None,run_id=None):
    '''Pertubs data and runs the specified clustering approach then gives an output of the homogeineity score plus\
    other meta info'''
    no_genes=len(genes)
    no_cells=len(cells)
    if run_id:
        out_report=('Processing',run_id,'fc:',fc,'gns:',no_genes,'cells',no_cells,discovery_method,modification_method,'component',pca_components)
        print(out_report)
        
        
    try:
        pertubed_data_dict=modification_in_subspace_for_selected_genes(inputdata=norm_indata,cells=cells,target_genes=genes,
                                                    fc=fc,gene_parameters_df=gene_parameters_df,all_genes_set=None,
                                                    genes_to_select_from_set=preselected_genes,
                                                    gene_modification_profile_df=gene_modification_profile_df)
        
        mod_gens = pertubed_data_dict['genes']
        
        perturbed_data_df=pertubed_data_dict['data']
        
        mod_cells =pertubed_data_dict['cells']
        
        #print(perturbed_data_df.loc[mod_genes[0:5],mod_cells[0:3]])
        
        matching_genes=pertubed_data_dict['matching_genes']
        
        var_genes_mat=filter_genes_dispersion(data=perturbed_data_df.transpose().values,flavor='seurat',min_disp=None,
                                              max_disp=None,min_mean=None, max_mean=None,n_bins=20,n_top_genes=nTopGenes,
                                              log=False,copy=False)
        
        var_genes_df=pd.DataFrame(var_genes_mat,index=perturbed_data_df.index)
        sig_var_genes_df=var_genes_df[var_genes_df.gene_subset==True]
        #var_genes_df=var_genes_df.sort_values(by=['dispersions_norm'],ascending=False)
        top_var_genes_df=var_genes_df[var_genes_df.gene_subset==True]
        top_var_genes_list=top_var_genes_df.index.tolist()
        top_genes_plus_all_mod_genes_list=top_var_genes_list[:]
        top_genes_plus_all_mod_genes_list.extend(mod_gens)
        top_genes_plus_all_mod_genes_set=set(top_genes_plus_all_mod_genes_list)
        mod_and_var_genes =set(mod_gens).intersection(top_var_genes_list)
        len_mod_and_var_genes=len(mod_and_var_genes) if not mod_and_var_genes is None else 0
        
        if post_mod_genes=='var_and_mod':
            
            perturbed_data_df=perturbed_data_df.loc[top_genes_plus_all_mod_genes_set,:]
            
        elif post_mod_genes=='mod':
            
            perturbed_data_df=perturbed_data_df.loc[mod_gens,:]
            
        elif post_mod_genes=='sig_and_mod':
            
            sig_top_var_genes_set=set(top_var_genes_list)
            
            sig_top_var_genes_set.update(mod_gens)
            
            perturbed_data_df=perturbed_data_df.loc[sig_top_var_genes_set,:]
        
        elif post_mod_genes=='sig_only':
            #sig_top_var_genes_set=set(var_genes_df[var_genes_df.gene_subset==True].index.tolist())
            perturbed_data_df=perturbed_data_df.loc[top_var_genes_list,:]
            
        elif post_mod_genes=='all':
            
            pass
       
        else:
            perturbed_data_df=perturbed_data_df.loc[top_var_genes_list,:]
        perturbed_data_df=utils.log_transform_dataframe(inputdata=perturbed_data_df,pseudo_value=pseudo_value)
        input_labs=['mod' if lb in mod_cells else 'unmod' for lb in perturbed_data_df.columns.tolist()]  
        if randomize_cells_labels==True:
            np.random.shuffle(input_labs)
        
        temp_res=cluster_data(inputdata=perturbed_data_df,labels=input_labs,transpose_inputdata=True,
                              discovery_method=discovery_method,pca_components=pca_components,
                              clustering_method=clustering_method,knn_neighbours=knn_neighbours,knn_radius=knn_radius,
                              knn_algorithm=knn_algorithm,knn_leaf_size=knn_leaf_size,knn_distance_metric=knn_distance_metric,
                              knn_p=knn_p,knn_cpus=knn_cpus,kmeans_clusters=kmeans_clusters,scale_kmeans=scale_kmeans,
                              tsne_components=tsne_components,tsne_perplexity=tsne_perplexity,corr_method=corr_method,
                              dbscan_min_cluster_size=dbscan_min_cluster_size,dbscan_min_samples=dbscan_min_samples,
                              dbscan_metric=dbscan_metric)
        out_list=temp_res[0]
        other_params=':'.join([str(par) for par in [quantiles,filter_genes_pattern,genes_and_cells_cut_off,normalize,
                                                    pseudo_value,find_var_genes,pca_components,clustering_method,
                                                    knn_neighbours,knn_radius,knn_algorithm,knn_leaf_size,knn_distance_metric,
                                                    knn_p,knn_cpus,kmeans_clusters,scale_kmeans,tsne_components,
                                                    tsne_perplexity,corr_method,
                                                    'genes_in_all_cells:'+str(genes_in_all_cells)] if par!=None])
        out_list.extend([no_cells, no_genes, fc, run_id, mod_cells, mod_gens,len_mod_and_var_genes,
                         modification_method, clustering_method,randomize_cells_labels,matching_genes, other_params])
        del temp_res   
        del pertubed_data_dict
        del var_genes_df
        del var_genes_mat
        del top_var_genes_df
        del top_var_genes_list
        del top_genes_plus_all_mod_genes_list
        del top_genes_plus_all_mod_genes_set
        del mod_and_var_genes
        del len_mod_and_var_genes
        return out_list
       
    except KeyboardInterrupt:
        
        return 'KeyboardInterrupt'
    except Exception as err:
        print("Unexpected error during the pertubation and var gene detecton!!", err)
        pass
    pass

#get_homogeneity_score_fh=open('get_homogeneity_score.log','w')
#@profile(stream=get_homogeneity_score_fh)
def get_homogeneity_score(input_array,labels,clustering_method='knn',knn_neighbours=3,
                          knn_radius=1.0,knn_algorithm='auto',knn_leaf_size=30,knn_distance_metric='minkowski',
                          knn_p=2,knn_cpus=1,kmeans_clusters=2,scale_kmeans=True,dbscan_min_cluster_size=5,
                          dbscan_min_samples=None,dbscan_metric='euclidean'):
    '''Runs the input data and generates the homogeneity score'''
    if clustering_method=='knn':
        knn_labels=run_knn(input_array=input_array,labels=labels,no_neighbours=knn_neighbours,
                           radius=knn_radius,algorithm=knn_algorithm,leaf_size=knn_leaf_size,
                           distance_metric=knn_distance_metric,p=knn_p,cpus=knn_cpus)
        hs = metrics.homogeneity_score(labels_true=labels,labels_pred=knn_labels)
        complet_score=metrics.completeness_score(labels_true=labels, labels_pred=knn_labels) 
        v_score=metrics.v_measure_score(labels_true=labels, labels_pred=knn_labels)
        adj_rand_index=metrics.adjusted_rand_score(labels_true=labels, labels_pred=knn_labels)
        adj_mutual_info=metrics.adjusted_mutual_info_score(labels_true=labels, labels_pred=knn_labels,
                                                           average_method='arithmetic')
        fm_score=metrics.fowlkes_mallows_score(labels_true=labels, labels_pred=knn_labels)
        euclidean_silhouette_score=metrics.silhouette_score(X=input_array, labels=knn_labels, metric='euclidean')
        cosine_silhouette_score=metrics.silhouette_score(X=input_array, labels=knn_labels, metric='cosine')
        calinski_harabaz_score=metrics.calinski_harabaz_score(X=input_array, labels=knn_labels)
        del input_array
        gc.collect()
        return {'hs':hs,'labs':knn_labels,'complet_score':complet_score,'v_score':v_score,'adj_rand_index':adj_rand_index,
               'adj_mutual_info':adj_mutual_info,'fm_score':fm_score,'calinski_harabaz_score':calinski_harabaz_score,
               'cosine_silhouette_score':cosine_silhouette_score,'euclidean_silhouette_score':euclidean_silhouette_score}
    if clustering_method=='kmeans':
        kmeans_result=run_kmeans(input_array=input_array,no_clusters=kmeans_clusters,scale_array=scale_kmeans)
        hs = metrics.homogeneity_score(labels_true=labels,labels_pred=kmeans_result.labels_)
        complet_score=metrics.completeness_score(labels_true=labels, labels_pred=kmeans_result.labels_) 
        v_score=metrics.v_measure_score(labels_true=labels, labels_pred=kmeans_result.labels_)
        adj_rand_index=metrics.adjusted_rand_score(labels_true=labels, labels_pred=kmeans_result.labels_)
        adj_mutual_info=metrics.adjusted_mutual_info_score(labels_true=labels, labels_pred=kmeans_result.labels_,
                                                           average_method='arithmetic')
        fm_score=metrics.fowlkes_mallows_score(labels_true=labels, labels_pred=kmeans_result.labels_)
        euclidean_silhouette_score=metrics.silhouette_score(X=input_array, labels=kmeans_result.labels_, metric='euclidean')
        cosine_silhouette_score=metrics.silhouette_score(X=input_array, labels=kmeans_result.labels_, metric='cosine')
        calinski_harabaz_score=metrics.calinski_harabaz_score(X=input_array, labels=kmeans_result.labels_)
        return {'hs':hs,'labs':kmeans_result.labels_,'complet_score':complet_score,'v_score':v_score,
                'adj_rand_index':adj_rand_index,'adj_mutual_info':adj_mutual_info,'fm_score':fm_score,
                'calinski_harabaz_score':calinski_harabaz_score,'cosine_silhouette_score':cosine_silhouette_score,
                'euclidean_silhouette_score':euclidean_silhouette_score}
    if clustering_method=='hdbscan':
        hdbscan_result=run_dbscan(input_array=input_array,in_min_cluster_size=dbscan_min_cluster_size,
                                  in_min_samples=dbscan_min_samples, in_metric=dbscan_metric)
        hs = metrics.homogeneity_score(labels_true=labels,labels_pred=hdbscan_result.labels_)
        complet_score=metrics.completeness_score(labels_true=labels, labels_pred=hdbscan_result.labels_) 
        v_score=metrics.v_measure_score(labels_true=labels, labels_pred=hdbscan_result.labels_)
        adj_rand_index=metrics.adjusted_rand_score(labels_true=labels, labels_pred=hdbscan_result.labels_)
        adj_mutual_info=metrics.adjusted_mutual_info_score(labels_true=labels, labels_pred=hdbscan_result.labels_,
                                                           average_method='arithmetic')
        fm_score=metrics.fowlkes_mallows_score(labels_true=labels, labels_pred=hdbscan_result.labels_)
        euclidean_silhouette_score=metrics.silhouette_score(X=input_array, labels=hdbscan_result.labels_, metric='euclidean')
        cosine_silhouette_score=metrics.silhouette_score(X=input_array, labels=hdbscan_result.labels_, metric='cosine')
        calinski_harabaz_score=metrics.calinski_harabaz_score(X=input_array, labels=hdbscan_result.labels_)
        #hs =0.0 if hs <0 else hs 
        return {'hs':hs,'labs':hdbscan_result.labels_,'complet_score':complet_score,'v_score':v_score,
                'adj_rand_index':adj_rand_index,'adj_mutual_info':adj_mutual_info,'fm_score':fm_score,
                'calinski_harabaz_score':calinski_harabaz_score,'cosine_silhouette_score':cosine_silhouette_score,
                'euclidean_silhouette_score':euclidean_silhouette_score}
    pass

def pc_pairs_homogeneity_scores_knn(pca_df,labels,neighbours=3,radius=1.0,algorithm='auto',
                                    leaf_size=30,distance_metric='minkowski',p=2,cpus=1):
    '''Get the PC components homogeneity scores for each pair based on knn classification'''
    pcs=list(pca_df.columns)
    pca_scores_dict={}
    labs_dict={}
    for iterat in itertools.combinations(pcs, 2):
        temp_pca=pca_df.loc[:,iterat]
        temp_knn_labs=run_knn(input_array=temp_pca,labels=labels,no_neighbours=neighbours,
                radius=radius,algorithm=algorithm,leaf_size=leaf_size,distance_metric=distance_metric,
                p=p,cpus=cpus)
        temp_hs_score=metrics.homogeneity_score(labels_pred=temp_knn_labs,labels_true=labels)
        pca_scores_dict['_'.join(iterat)]=temp_hs_score
        labs_dict['_'.join(iterat)]=temp_knn_labs
    return {'pc_scores':pca_scores_dict,'labs':labs_dict}



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








    
