import os
import argparse
import datetime

import pandas as pd
import numpy as np

from . import hgmd
from . import visualize as vis
from . import quads
import sys
import multiprocessing
import time
import math
import matplotlib.pyplot as plt
import random
#from docs.source import conf


def init_parser(parser):
    """Initialize parser args."""
    parser.add_argument(
        'marker', type=str,
        help=("Marker file input")
    )
    parser.add_argument(
        'tsne', type=str,
        help=("tsne file input")
    )
    parser.add_argument(
        'cluster', type=str,
        help=("Cluster file input")
    )
    parser.add_argument(
        '-g', nargs='?', default=None,
        help="Optional Gene list"
    )
    parser.add_argument(
        'output_path', type=str,
        help="the output directory where output files should go"
    )
    parser.add_argument(
        '-C', nargs='?', default=None,
        help="Num of cores avail for parallelization"
    )
    parser.add_argument(
        '-X', nargs='?', default=None,
        help="X argument for XL-mHG"
    )
    parser.add_argument(
        '-L', nargs='?', default=None,
        help="L argument for XL-mHG"
    )
    parser.add_argument(
        '-Abbrev', nargs='?',default=False,
        help="Choose between abbreviated or full 3-gene computation"
    )
    parser.add_argument(
        '-K', nargs='?',default=None,
        help="K-gene combinations to include"
    )
    parser.add_argument(
        '-Down', nargs='?',default=False,
        help="Downsample"
    )
    return parser


def read_data(cls_path, tsne_path, marker_path, gene_path, D):
    """
    Reads in cluster series, tsne data, marker expression without complements
    at given paths.
    """
    
    cls_ser = pd.read_csv(
        cls_path, sep='\t', index_col=0, names=['cell', 'cluster'], squeeze=True
    )
    tsne = pd.read_csv(
        tsne_path, sep='\t', index_col=0, names=['cell', 'tSNE_1', 'tSNE_2']
    )
    #could be optimized to read and check against gene list simultaneously
    #if this is being a bottleneck. Would require unboxing pd.read_csv though.
    start_= time.time()
    no_complement_marker_exp = pd.read_csv(
        marker_path,sep='\t', index_col=0
        ).rename_axis('cell',axis=1)
    #gene list filtering
    #print(no_complement_marker_exp)
    no_complement_marker_exp = np.transpose(no_complement_marker_exp)

    #gene filtering
    #-------------#
    if gene_path is None:
        pass
    else:
        with open(gene_path, "r") as genes:
            init_read = genes.read().splitlines()
            master_str = str.upper(init_read[0])
            master_gene_list = master_str.split(",")
            for column_name in no_complement_marker_exp.columns:
                if str.upper(column_name) in master_gene_list:
                    pass
                else:
                    no_complement_marker_exp.drop(column_name, axis=1, inplace=True)
                
    #-------------#

    #downsampling
    #-------------#
    if D is False:
        pass
    else:
        #total number of cells input
        N = len(cls_ser)
        #print(N)
        #downsample target
        M = int(D)
        if N <= M:
            return (cls_ser, tsne, no_complement_marker_exp, gene_path)
        clusters = sorted(cls_ser.unique())
        counts = { x : 0 for x in clusters}
        for clus in cls_ser:
            counts[clus] = counts[clus]+1
        #at this point counts has values for # cells in cls
        #dict goes like ->{ cluster:#cells }
        take_nums = {x : 0 for x in clusters}
        for clstr in take_nums:
            take_nums[clstr] = math.ceil(counts[clstr]*(M/N))
        summ = 0
        for key in take_nums:
            summ = summ + take_nums[key]
        #print('Downsampled cell num ' + str(summ))
        counts= { x : 0 for x in clusters}
        new_cls_ser = cls_ser.copy(deep=True)
        keep_first = 0
        for index,value in new_cls_ser.iteritems():
            keep_first = keep_first + 1
            if keep_first ==1:
                counts[value] = counts[value]+1
                continue
            new_cls_ser.drop(index,inplace=True)
        cls_ser.drop(cls_ser.index[0],inplace=True)
        #Now new_cls_ser has all removed except first item, which we can keep
        for num in range(N-1):
            init_rand_num = random.randint(0,N-num-1-1)
            if counts[cls_ser[init_rand_num]] >= take_nums[cls_ser[init_rand_num]]:
                cls_ser.drop(cls_ser.index[init_rand_num], inplace=True)
                continue
            new_cls_ser = new_cls_ser.append(pd.Series([cls_ser[init_rand_num]], index=[cls_ser.index[init_rand_num]]))
            counts[cls_ser[init_rand_num]] = counts[cls_ser[init_rand_num]]+1
            cls_ser.drop(cls_ser.index[init_rand_num], inplace=True)
            
        new_cls_ser.rename_axis('cell',inplace=True)
        new_cls_ser.rename('cluster', inplace=True)
        return(new_cls_ser,tsne,no_complement_marker_exp,gene_path)


    #-------------#

            
    return (cls_ser, tsne, no_complement_marker_exp, gene_path)

def process(cls,X,L,plot_pages,cls_ser,tsne,marker_exp,gene_file,csv_path,vis_path,pickle_path,cluster_number,K,abbrev):
    #for cls in clusters:
    # To understand the flow of this section, read the print statements.
    start_cls_time = time.time()
    print('########\n# Processing cluster ' + str(cls) + '...\n########')
    print(str(K) + ' gene combinations')
    print('Running t test on singletons...')
    t_test = hgmd.batch_t(marker_exp, cls_ser, cls)
    print('Calculating fold change')
    fc_test = hgmd.batch_fold_change(marker_exp, cls_ser, cls)
    print('Running XL-mHG on singletons...')
    xlmhg = hgmd.batch_xlmhg(marker_exp, cls_ser, cls, X=X, L=L)
    # We need to slide the cutoff indices before using them,
    # to be sure they can be used in the real world. See hgmd.mhg_slide()
    cutoff_value = hgmd.mhg_cutoff_value(
        marker_exp, xlmhg[['gene_1', 'mHG_cutoff']]
    )
    xlmhg = xlmhg[['gene_1', 'HG_stat', 'mHG_pval']].merge(
        hgmd.mhg_slide(marker_exp, cutoff_value), on='gene_1'
    )
    # Update cutoff_value after sliding
    cutoff_value = pd.Series(
        xlmhg['cutoff_val'].values, index=xlmhg['gene_1']
    )
    xlmhg = xlmhg\
        .sort_values(by='HG_stat', ascending=True)

    ###########
    #ABBREVIATED 
    if abbrev == 'True':
        print('Heuristic Abbreviation initiated')
        count = 0
        trips_list=[]
        for index,row in xlmhg.iterrows():
            if count == 100:
                break
            else:
                trips_list.append(row[0])
                count = count + 1
    else:
        trips_list = None
    ############
    print('Creating discrete expression matrix...')
    discrete_exp = hgmd.discrete_exp(marker_exp, cutoff_value,trips_list)

    discrete_exp_full = discrete_exp.copy()
    print('Finding simple true positives/negatives for singletons...')
    #Gives us the singleton TP/TNs for COI and for rest of clusters
    #COI is just a DF, rest of clusters are a dict of DFs
    (sing_tp_tn, other_sing_tp_tn) = hgmd.tp_tn(discrete_exp, cls_ser, cls, cluster_number)

    '''
    for key in other_sing_tp_tn:
        for index,row in other_sing_tp_tn[key].iterrows():
            if index == 'LY6D':
                print(index)
                print('Cluster '+str(key))
                print('TP = ')
                print(row[0])
            if index == 'CD3G_c':
                print(index)
                print('Cluster '+str(key))
                print('TP = ')
                print(row[0])
                
    time.sleep(10000) 
    '''
    print('Finding pair expression matrix...')
    (
        gene_map, in_cls_count, pop_count,
        in_cls_product, total_product, upper_tri_indices,
        cluster_exp_matrices, cls_counts
    ) = hgmd.pair_product(discrete_exp, cls_ser, cls, cluster_number,trips_list)


    if K >= 4:
        print('')
        print('Starting quads')
        print('')
        quads_in_cls, quads_total, quads_indices, odd_gene_mapped, even_gene_mapped = quads.combination_product(discrete_exp,cls_ser,cls,trips_list)
        print('')
        print('')
        print('')
        print('')
        print('HG TEST ON QUADS')
        quads_fin = quads.quads_hg(gene_map,in_cls_count,pop_count,quads_in_cls,quads_total,quads_indices,odd_gene_mapped,even_gene_mapped)





    
    if K == 3:
        start_trips = time.time()
        print('Finding Trips expression matrix...')
        trips_in_cls,trips_total,trips_indices,gene_1_mapped,gene_2_mapped,gene_3_mapped = hgmd.combination_product(discrete_exp,cls_ser,cls,trips_list)
        end_trips = time.time()
        print(str(end_trips-start_trips) + ' seconds')

    HG_start = time.time()
    print('Running hypergeometric test on pairs...')
    
    pair, revised_indices = hgmd.pair_hg(
        gene_map, in_cls_count, pop_count,
        in_cls_product, total_product, upper_tri_indices, abbrev
    )

    HG_end = time.time()
    print(str(HG_end-HG_start) + ' seconds')
    
    pair_out_initial = pair\
    .sort_values(by='HG_stat', ascending=True)
    
    pair_out_initial['rank'] = pair_out_initial.reset_index().index + 1
    pair_out_initial.to_csv(
    csv_path + '/cluster_' + str(cls) + '_pair_init_rank.csv'
    )

    if K == 3:
        HG_start = time.time()
        print('Running hypergeometric test & TP/TN on trips...')
        if abbrev == 'True':
            trips = hgmd.trips_hg(
                gene_map,in_cls_count,pop_count,
                trips_in_cls,trips_total,trips_indices,
                gene_1_mapped,gene_2_mapped,gene_3_mapped
                )
        else:
            trips = hgmd.trips_hg_full(
                gene_map,in_cls_count,pop_count,
                trips_in_cls,trips_total,trips_indices,
                gene_1_mapped,gene_2_mapped,gene_3_mapped
                )
            #print(trips)
            HG_end = time.time()
            print(str(HG_end-HG_start) + ' seconds')



    # Pair TP/TN FOR THIS CLUSTER
    print('Finding simple true positives/negatives for pairs...')
    pair_tp_tn = hgmd.pair_tp_tn(
        gene_map, in_cls_count, pop_count,
        in_cls_product, total_product, upper_tri_indices, abbrev, revised_indices
    )
    #accumulates pair TP/TN vals for all other clusters
    ##NEW
    other_pair_tp_tn = {}
    for key in cluster_exp_matrices:
        new_pair_tp_tn = hgmd.pair_tp_tn(
            gene_map, cls_counts[key], pop_count,
            cluster_exp_matrices[key], total_product, upper_tri_indices,
            abbrev, revised_indices
        )
        other_pair_tp_tn[key] = new_pair_tp_tn
        other_pair_tp_tn[key].set_index(['gene_1','gene_2'],inplace=True)
    pair = pair\
        .merge(pair_tp_tn, on=['gene_1', 'gene_2'], how='left')
    pair_tp_tn.set_index(['gene_1','gene_2'],inplace=True)
    sing_tp_tn.set_index(['gene_1'], inplace=True)
    rank_start = time.time()

    '''
    print(other_pair_tp_tn)
    for key in other_pair_tp_tn:
        for index, row in other_pair_tp_tn[key].iterrows():
            if 'LY6D' in index and 'CD3G_c' in index:
                print(index)
                print('cluster ' + str(key))
                print('TP = ')
                print(row[0])
    time.sleep(10000)
    '''
    print('Finding NEW Rank')
    ranked_pair,histogram = hgmd.ranker(pair,xlmhg,sing_tp_tn,other_sing_tp_tn,other_pair_tp_tn,cls_counts,in_cls_count)
    rank_end = time.time()
    print(str(rank_end - rank_start) + ' seconds')

    
    
    # Save TP/TN values to be used for non-cluster-specific things
    print('Pickling data for later...')
    sing_tp_tn.to_pickle(pickle_path + 'sing_tp_tn_' + str(cls))
    pair_tp_tn.to_pickle(pickle_path + 'pair_tp_tn_' + str(cls))
    #trips_tp_tn.to_pickle(pickle_path + 'trips_tp_tn' + str(cls))
    print('Exporting cluster ' + str(cls) + ' output to CSV...')
    sing_output = xlmhg\
        .merge(t_test, on='gene_1')\
        .merge(fc_test, on='gene_1')\
        .merge(sing_tp_tn, on='gene_1')\
        .set_index('gene_1')\
        .sort_values(by='HG_stat', ascending=True)

    sing_output['hgrank'] = sing_output.reset_index().index + 1
    sing_output.sort_values(by='FoldChange', ascending=False, inplace=True)
    sing_output['fcrank'] = sing_output.reset_index().index + 1
    sing_output['finrank'] = sing_output[['hgrank', 'fcrank']].mean(axis=1)
    sing_output.sort_values(by='finrank',ascending=True,inplace=True)
    sing_output['rank'] = sing_output.reset_index().index + 1
    sing_output.drop('finrank',axis=1, inplace=True)
    count = 1
    for index,row in sing_output.iterrows():
        if count == 100:
            break
        if row[0] >= .05:
            sing_output.at[index,'Plot'] = 0
        else:
            sing_output.at[index,'Plot'] = 1
            count = count + 1
    sing_output.to_csv(
        csv_path + '/cluster_' + str(cls) + '_singleton.csv'
    )
    sing_stripped = sing_output[
        ['HG_stat', 'TP', 'TN']
    ].reset_index().rename(index=str, columns={'gene_1': 'gene_1'})
    
    ##########
    #change to reflect ranked_pair
    ##########
    #print('yabada')
    #print(pair)
    #pair_output = ranked_pair\
    #    .merge(pair_tp_tn, on=['gene_1', 'gene_2'], how='left')
    #print(pair)
    #pair_output['rank'] = pair_output.reset_index().index + 1
    ranked_pair.to_csv(
        csv_path + '/cluster_' + str(cls) + '_pair_ranked.csv'
    )
    #Add trips data pages
    #does not currently do new rank scheme
    if K == 3:
        trips_output = trips\
          .sort_values(by='HG_stat', ascending=True)
          #print(trips_output)
        trips_output['rank'] = trips_output.reset_index().index + 1
        trips_output.to_csv(
            csv_path + '/cluster_' + str(cls) + '_trips.csv'
            )
    else:
        trips_output = int(1)
    if K >= 4:
        quads_final = quads_fin\
          .sort_values(by='HG_stat', ascending=True)
        quads_final['rank'] = quads_final.reset_index().index + 1
        quads_final.to_csv(
            csv_path + '/cluster_' + str(cls) + '_quads.csv'
            )
    else:
        quads_final = int(1)
    print('Drawing plots...')
    #plt.bar(list(histogram.keys()), histogram.values(), color='b')
    #plt.savefig(vis_path + '/cluster_' + str(cls) + '_pair_histogram')

    #if cls == fincls:
    # cls = 0
    vis.make_plots(
        pair=ranked_pair,
        sing=sing_output,
        trips=trips_output,
        quads_fin=quads_final,
        tsne=tsne,
        discrete_exp=discrete_exp_full,
        marker_exp=marker_exp,
        plot_pages=plot_pages,
        combined_path=vis_path + '/cluster_' + str(cls) + '_combined.pdf',
        sing_combined_path=vis_path + '/cluster_' +
        str(cls) + '_singleton_combined.pdf',
        discrete_path=vis_path + '/cluster_' + str(cls) + '_discrete.pdf',
        tptn_path=vis_path + 'cluster_' + str(cls) + '_TP_TN.pdf',
        trips_path=vis_path + 'cluster_' + str(cls) + '_discrete_trios.pdf',
        quads_path=vis_path + 'cluster_' + str(cls) + '_discrete_quads.pdf',
        sing_tptn_path=vis_path + 'cluster_' + str(cls) + '_singleton_TP_TN.pdf'
        )
    end_cls_time=time.time()
    print(str(end_cls_time - start_cls_time) + ' seconds')
    #time.sleep(10000)


def main():
    """Hypergeometric marker detection. Finds markers identifying a cluster.

    Reads in data from single-cell RNA sequencing. Data is in the form of 3
    CSVs: gene expression data by gene by cell, 2-D tSNE data by cell, and the
    clusters of interest by cell. Creates a list of genes and a list of gene
    pairs (including complements), ranked by hypergeometric and t-test
    significance. The highest ranked marker genes generally best identify the
    cluster of interest. Saves these lists to CSV and creates gene expression
    visualizations.
    """
    # TODO: more precise description

    start_dt = datetime.datetime.now()
    start_time = time.time()
    print("Started on " + str(start_dt.isoformat()))
    args = init_parser(argparse.ArgumentParser(
        description=("Hypergeometric marker detection. Finds markers identifying a cluster. Documentation available at https://hgmd.readthedocs.io/en/latest/index.html")
    )).parse_args()
    output_path = args.output_path
    C = args.C
    K = args.K
    Abbrev = args.Abbrev
    Down = args.Down
    X = args.X
    L = args.L
    marker_file = args.marker
    tsne_file = args.tsne
    cluster_file = args.cluster
    gene_file = args.g
    plot_pages = 30  # number of genes to plot (starting with highest ranked)

    # TODO: gene pairs with expression ratio within the cluster of interest
    # under [min_exp_ratio] were ignored in hypergeometric testing. This
    # functionality is currently unimplemented.
    # min_exp_ratio = 0.4

    csv_path = output_path + 'data/'
    vis_path = output_path + 'vis/'
    pickle_path = output_path + '_pickles/'
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(pickle_path, exist_ok=True)
    if C is not None:
        C = abs(int(C))
    else:
        C = 1
    if X is not None:
        X = int(X)
        print("Set X to " + str(X) + ".")
    if L is not None:
        L = int(L)
        print("Set L to " + str(L) + ".")
    if K is not None:
        K = int(K)
    else:
        K = 2
    if K > 4:
        K = 4
        print('Only supports up to 4-gene combinations currently, setting K to 4')
    print("Reading data...")
    if gene_file is None:
        (cls_ser, tsne, no_complement_marker_exp, gene_path) = read_data(
            cls_path=cluster_file,
            tsne_path=tsne_file,
            marker_path=marker_file,
            gene_path=None,
            D=Down
        )
    else:
        (cls_ser, tsne, no_complement_marker_exp, gene_path) = read_data(
            cls_path=cluster_file,
            tsne_path=tsne_file,
            marker_path=marker_file,
            gene_path=gene_file,
            D=Down
        )
    print("Generating complement data...")
    marker_exp = hgmd.add_complements(no_complement_marker_exp)
    #throw out vals that show up in expression matrix but not in cluster assignments
    for index,row in marker_exp.iterrows():
        if index in cls_ser.index.values.tolist():
            continue
        else:
            marker_exp.drop(index, inplace=True)
    
    # Process clusters sequentially
    clusters = cls_ser.unique()
    clusters.sort()
    
    if clusters[0] == 0:
        cluster_change = np.delete(clusters,0)
        cluster_change_2 = np.append(cluster_change,cluster_change[-1]+1)
        clusters = cluster_change_2
        fin_clus = clusters[-1]
        cls_ser.replace(0,fin_clus,inplace=True)
            #print(row)
    #Below could probably be optimized a little (new_clust not necessary),
    #instead of new clust just go from (x-1)n to (x)n in clusters
    #but it works for now and the complexity it adds is trivial and it makes debugging easier
    #cores is number of simultaneous threads you want to run, can be set at will
    cores = C
    cluster_number = len(clusters)
    # if core number is bigger than number of clusters, just set it equal to number of clusters
    if cores > len(clusters):
        cores = len(clusters)
    #below loops allow for splitting the job based on core choice
    group_num  = math.ceil((len(clusters) / cores ))
    for element in range(group_num):
        new_clusters = clusters[:cores]
        print(new_clusters)
        jobs = []
        #this loop spawns the workers and runs the code for each assigned.
        #workers assigned based on the new_clusters list which is the old clusters
        #split up based on core number e.g.
        #clusters = [1 2 3 4 5 6] & cores = 4 --> new_clusters = [1 2 3 4], new_clusters = [5 6]
        for cls in new_clusters:
            p = multiprocessing.Process(target=process,
                args=(cls,X,L,plot_pages,cls_ser,tsne,marker_exp,gene_file,csv_path,vis_path,pickle_path,cluster_number,K,Abbrev))
            jobs.append(p)
            p.start()
        p.join()
        
        new_clusters = []
        clusters = clusters[cores:len(clusters)]
        print(clusters)

    end_time = time.time()

    

    # Add text file to keep track of everything
    end_dt = datetime.datetime.now()
    print("Ended on " + end_dt.isoformat())
    metadata = open(output_path + 'metadata.txt', 'w')
    metadata.write("Started: " + start_dt.isoformat())
    metadata.write("\nEnded: " + end_dt.isoformat())
    metadata.write("\nElapsed: " + str(end_dt - start_dt))
    #metadata.write("\nGenerated by COMET version " + conf.version)


    print('Took ' + str(end_time-start_time) + ' seconds')
    print('Which is ' + str( (end_time-start_time)/60 ) + ' minutes')

if __name__ == '__main__':
    main()
