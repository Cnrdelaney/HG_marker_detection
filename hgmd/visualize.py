import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from itertools import repeat
import time

"""
Set of modularized visualization functions for COMET; producing graphs in PDFs.
"""

"""
Goals: Separate singleton and pair graphs for:

    - TP/TN with gene names

    - Side-by-side comparison between discrete and continuous expression

For singletons and pairs combined:

    - Side-by-side comparison of single gene discrete expression and resulting
      pair discrete expression.
"""


CMAP_CONTINUOUS = cm.get_cmap('nipy_spectral')
CMAP_DISCRETE = cm.get_cmap('bwr')


def make_plots(
    pair, sing, trips, tsne, discrete_exp, marker_exp, plot_pages,
    combined_path, sing_combined_path, discrete_path, tptn_path,trips_path,sing_tptn_path
):
    """
    General function for all visualization generation.  Arguments should be
    self-explanatory.  See __main__.
    """
    # cutoff maps genes to their (absolute) cutoff. I.e. complements are mapped
    # to positive cutoffs.
    #print(discrete_exp)
    #print(marker_exp)
    cutoff = sing['cutoff_val'].abs()
    rank = sing['rank']
    
    pair_sing_only = pair[pair['gene_2'].isnull()]
    sing_rank = pd.Series(
        pair_sing_only['rank'].values, index=pair_sing_only['gene_1']
    )

    p_short = pair[pair['Plot']==1]
    #p_short = pair.iloc[:plot_pages]
    s_short = sing.iloc[:plot_pages]
    t_short = trips.iloc[:plot_pages]


    vmt = np.vectorize(make_title)
    d_plot_genes = zip(
        zip(
            vmt(
                p_short['gene_1'], p_short['gene_2'],
                p_short['rank'], p_short['gene_1'].map(cutoff)
            ), vmt(
                p_short['gene_1'], np.nan,
                p_short['gene_1'].map(rank),
                p_short['gene_1'].map(cutoff)
            ), vmt(
                p_short['gene_2'], np.nan,
                p_short['gene_2'].map(rank),
                p_short['gene_2'].map(cutoff)
            )
        ), p_short['gene_1'].values, p_short['gene_2'].values
    )

    
    vmt_2 = np.vectorize(make_trips_title)
    t_plot_genes = zip(
        zip(
            vmt_2(
                t_short['gene_1'], t_short['gene_2'], t_short['gene_3'],
                t_short['rank'], t_short['gene_1'].map(cutoff)
            ), vmt_2(
                t_short['gene_1'], np.nan, np.nan,
                t_short['gene_1'].map(rank),
                t_short['gene_1'].map(cutoff)
            ), vmt_2(
                t_short['gene_2'], np.nan, np.nan,
                t_short['gene_2'].map(rank),
                t_short['gene_2'].map(cutoff)
            ), vmt_2(
                t_short['gene_3'], np.nan, np.nan,
                t_short['gene_3'].map(rank),
                t_short['gene_3'].map(cutoff)
            )
        ), t_short['gene_1'].values, t_short['gene_2'].values, t_short['gene_3'].values
    )
    

    print("Drawing discrete plots for pairs...")
    make_discrete_plots(
        tsne, discrete_exp, d_plot_genes, discrete_path, 2
    )
    print("Drawing discrete plots for trips...")
    make_discrete_plots(
        tsne, discrete_exp, t_plot_genes, trips_path, 3
    )
    

    
    c_plot_genes = zip(
        zip(
            vmt(
                p_short['gene_1'], np.nan,
                p_short['gene_1'].map(rank),
                p_short['gene_1'].map(cutoff)
            ), vmt(
                p_short['gene_2'], np.nan,
                p_short['gene_2'].map(rank),
                p_short['gene_2'].map(cutoff)
            )
        ), p_short['gene_1'].values, p_short['gene_2'].values
    )
    print("Drawing combined plots...")
    make_combined_plots(
        tsne, discrete_exp, marker_exp, c_plot_genes, combined_path
    )
    c_s_plot_genes = zip(
        zip(
            vmt(
                s_short.index, np.nan,
                s_short['rank'], s_short['cutoff_val']
            ), repeat(np.nan)
        ), s_short.index, repeat(np.nan)
    )
    print("Drawing singleton combined plots...")
    make_combined_plots(
        tsne, discrete_exp, marker_exp, c_s_plot_genes, sing_combined_path
    )
    pair_tp_tn = pair[['gene_1', 'gene_2', 'TP', 'TN']]
    sing_tp_tn = sing[['TP', 'TN']]
    print("Drawing true positive/negative plots...")
    make_tp_tn_plot(
        zip(p_short['gene_1'], p_short['gene_2']),
        sing_tp_tn, pair_tp_tn, tptn_path
    )
    make_tp_tn_plot(
        zip(s_short.index, repeat(np.nan)),
        sing_tp_tn, pair_tp_tn, sing_tptn_path
    )


def make_title(gene_1, gene_2, rank, cutoff_val):
    """Makes a plot title for a gene or gene pair.

    Formatting: for pairs, 'rank $rank: $gene_1+$gene_2', and for singletons,
    'rank $rank: $gene_1 $cutoff_val'.  gene_2 should be None for singletons.

    :param genes: A DataFrame with columns 'gene_1', 'gene_2', 'rank',
        'cutoff_val'.

    :returns: A list of strings containing the titles, with indices
              corresponding to the input DataFrame.

    :rtype: string list
    """

    if pd.isna(gene_2):
        return ("rank %.0f: %s %.3f" % (rank, gene_1, cutoff_val))
    else:
        return ("rank %.0f: %s+%s" % (rank, gene_1, gene_2))

def make_trips_title(gene_1,gene_2,gene_3,rank,cutoff_val):
    
    if pd.isna(gene_2):
        return ("rank %.0f: %s %.3f" % (rank, gene_1, cutoff_val))
    elif pd.isna(gene_3):
        return ("rank %.0f: %s+%s" % (rank, gene_1, gene_2))
    else:
        return ("rank %.0f: %s+%s+%s" % (rank, gene_1, gene_2, gene_3)) 

def make_plot(ax, title, coords, cmap, draw_cbar=False):
    """
    Make a single graph on ax with given specs.  Plots only absolute values.
    """
    ax.set_title(title)
    ax.set_xlabel('tSNE_1')
    ax.set_ylabel('tSNE_2')
    sc = ax.scatter(
        x=coords[0],
        y=coords[1],
        c=abs(coords[2]),
        s=2,
        cmap=cmap
    )
    if draw_cbar:
        plt.colorbar(sc, ax=ax)


def make_discrete_plots(tsne, discrete_exp, plot_genes, path,num):
    """Plots discrete gene expression of paired genes to PDF.

    For each gene pair listed in plot_genes, make three scatterplots.  First, a
    plot showing joint expression.  Then, two plots showing singleton
    expression for each genes.  If a single gene is passed, plot only its
    expression, and make two blank plots.  Save each gene/gene pair as a PDF
    page, then save to path.

    :param tsne: A DataFrame with 'cell', 'tSNE_1', and 'tSNE_2' columns.
    :param discrete_exp: A DataFrame whose rows are cell identifiers, columns
        are gene identifiers, and values are boolean values representing gene
        expression.
    :param plot_genes: A list of 3-tuples, where the first element of each
        tuple is another 3-tuple containing the three plot titles to be used.
        The other 2 elements are the gene names to be plotted.
    :param path: The path to which the PDF will be saved.

    :returns: Nothing.
    """
    def make_trips_discrete_page(fig, ax_triple, titles, gene_1, gene_2, gene_3):
        """Make page with trips discrete plots given titles and genes."""
        coords_df = tsne.merge(discrete_exp[[gene_1, gene_2, gene_3]], on='cell')
        coords_df['pair'] = coords_df[gene_1] * coords_df[gene_2]
        coords_df['trips'] = coords_df[gene_1] * coords_df[gene_2] * coords_df[gene_3]

        for (graph_index, z_label) in ((0, 'trips'), (1, gene_1), (2, gene_2), (3, gene_3)):
            make_plot(
                ax=ax_triple[graph_index],
                title=titles[graph_index],
                coords=(
                    coords_df['tSNE_1'].values,
                    coords_df['tSNE_2'].values,
                    coords_df[z_label].values
                ),
                cmap=CMAP_DISCRETE
            )
        
    def make_pair_discrete_page(fig, ax_triple, titles, gene_1, gene_2):
        """Make page with three discrete plots given titles and genes."""
        coords_df = tsne.merge(discrete_exp[[gene_1, gene_2]], on='cell')
        coords_df['pair'] = coords_df[gene_1] * coords_df[gene_2]

        for (graph_index, z_label) in ((0, 'pair'), (1, gene_1), (2, gene_2)):
            make_plot(
                ax=ax_triple[graph_index],
                title=titles[graph_index],
                coords=(
                    coords_df['tSNE_1'].values,
                    coords_df['tSNE_2'].values,
                    coords_df[z_label].values
                ),
                cmap=CMAP_DISCRETE
            )

    def make_single_discrete_page(fig, ax_triple, title, gene):
        """Make page with one discrete plot given title and gene"""
        print(title)
        coords_df = tsne.merge(discrete_exp[[gene]], on='cell')
        make_plot(
            ax=ax_triple[0],
            title=title[0],
            coords=(
                coords_df['tSNE_1'].values,
                coords_df['tSNE_2'].values,
                coords_df[gene].values
            ),
            cmap=CMAP_DISCRETE
        )

    with PdfPages(path) as pdf:
        for plot_gene in plot_genes:
            # print(plot_gene)
            if num == 3:
                fig, ax_triple = plt.subplots(ncols=4, figsize=(15, 5))
                if pd.isnull(plot_gene[2]):
                    make_single_discrete_page(
                        fig=fig, ax_triple=ax_triple,
                        title=plot_gene[0],
                        gene=plot_gene[1]
                        )
                elif pd.isnull(plot_gene[3]):
                    make_pair_discrete_page(
                        fig=fig, ax_triple=ax_triple,
                        titles=plot_gene[0],
                        gene_1=plot_gene[1],
                        gene_2=plot_gene[2]
                        )
                else:
                    make_trips_discrete_page(
                        fig=fig, ax_triple=ax_triple,
                        titles=plot_gene[0],
                        gene_1=plot_gene[1],
                        gene_2=plot_gene[2],
                        gene_3=plot_gene[3]
                        )
            else:
                fig, ax_triple = plt.subplots(ncols=3, figsize=(15, 5))
                if pd.isnull(plot_gene[2]):
                    make_single_discrete_page(
                        fig=fig, ax_triple=ax_triple,
                        title=plot_gene[0],
                        gene=plot_gene[1]
                        )
                else:
                    make_pair_discrete_page(
                        fig=fig, ax_triple=ax_triple,
                        titles=plot_gene[0],
                        gene_1=plot_gene[1],
                        gene_2=plot_gene[2]
                        )
                
            pdf.savefig(fig)
            plt.close(fig)

def make_combined_plots(tsne, discrete_exp, marker_exp, plot_genes, path):
    """Plots discrete alongside continuous expression to PDF.

    For each gene/gene pair listed in plot_genes, make two scatterplots: a plot
    showing discrete expression, and a plot showing expression on a color
    spectrum.  For gene pairs, make these two plots separately for each gene.
    Save each gene/gene pair as a PDF page, then save to path.

    :param tsne: A DataFrame with 'cell', 'tSNE_1', and 'tSNE_2' columns.
    :param discrete_exp: A DataFrame whose rows are cell identifiers, columns
        are gene identifiers, and values are boolean values representing gene
        expression.
    :param marker_exp: A DataFrame whose rows are cell identifiers, columns are
        gene identifiers, and values are float values representing gene
        expression.
    :param plot_genes: A list of 3-tuples, where the first element of each
        tuple is another 2-tuple containing the two titles to be used..
        The other 2 elements are the gene names to be plotted.
    :param path: The path to which the PDF will be saved.

    :returns: Nothing.
    """
    def make_pair_combined_page(fig, axes, titles, gene_1, gene_2):
        """Make page with two pairs of plots for given genes."""
        disc_coords = tsne.merge(discrete_exp[[gene_1, gene_2]], on='cell')
        cont_coords = tsne.merge(marker_exp[[gene_1, gene_2]], on='cell')
        for (graph_index, z_label) in ((0, gene_1), (1, gene_2)):
            make_plot(
                ax=axes[graph_index][0], title=titles[graph_index],
                coords=(
                    disc_coords['tSNE_1'].values,
                    disc_coords['tSNE_2'].values,
                    disc_coords[z_label].values
                ),
                cmap=CMAP_DISCRETE
            )
            make_plot(
                ax=axes[graph_index][1], title=str(z_label),
                coords=(
                    cont_coords['tSNE_1'].values,
                    cont_coords['tSNE_2'].values,
                    cont_coords[z_label].values
                ),
                cmap=CMAP_CONTINUOUS, draw_cbar=True
            )

    def make_single_combined_page(fig, title, axes, gene):
        """Make page with single pair of plot of given gene."""
        disc_coords = tsne.merge(discrete_exp[[gene]], on='cell')
        cont_coords = tsne.merge(marker_exp[[gene]], on='cell')
        make_plot(
            ax=axes[0], title=title,
            coords=(
                disc_coords['tSNE_1'].values,
                disc_coords['tSNE_2'].values,
                disc_coords[gene].values
            ),
            cmap=CMAP_DISCRETE
        )
        make_plot(
            ax=axes[1], title=str(gene),
            coords=(
                cont_coords['tSNE_1'].values,
                cont_coords['tSNE_2'].values,
                cont_coords[gene].values
            ),
            cmap=CMAP_CONTINUOUS, draw_cbar=True
        )

    with PdfPages(path) as pdf:
        for plot_gene in plot_genes:
            if pd.isnull(plot_gene[2]):
                fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
                make_single_combined_page(
                    fig, plot_gene[0][0], axes, plot_gene[1]
                )
            else:
                fig, axes = plt.subplots(
                    nrows=2, ncols=2, figsize=(10, 10)
                )
                make_pair_combined_page(
                    fig, axes, plot_gene[0], plot_gene[1], plot_gene[2]
                )
            pdf.savefig(fig)
            plt.close(fig)


def make_tp_tn_plot(plot_genes, sing_tp_tn, pair_tp_tn, path):
    """Plots TP/TN rates of genes/pairs to PDF.

    For each gene/gene pair listed in plot_genes, plot their TP/TN rate on a
    scatterplot, labeling the point with the gene/gene pair name.  When done,
    output this scatterplot to PDF and save to path.

    :param plot_genes: An array whose elements are tuples representing gene
        pairs.  If the second element is empty, it represents a singleton.
    :param sing_tp_tn: A DataFrame with ''TP', and 'TN' columns, with gene
        indices
    :param pair_tp_tn: A DataFrame with 'gene_1', 'gene_2', 'TP', and 'TN'
        columns.
    :param path: The path to which the PDF will be saved.

    :returns: Nothing.
    """
    PADDING = 0.002

    fig = plt.figure(figsize=[15, 15])
    plt.xlabel("True positive")
    plt.ylabel("True negative")
    plt.title("True positive/negative")
    plt.axis([0.0, 1.1, 0.0, 1.1])

    def get_data(genes):
        if pd.isnull(genes[1]):
            title = genes[0]
            data_row = sing_tp_tn.loc[genes[0]]
        else:
            title = genes[0] + "+" + genes[1]
            data_row = pair_tp_tn[
                ((pair_tp_tn['gene_1'] == genes[0]) &
                 (pair_tp_tn['gene_2'] == genes[1]))
                | ((pair_tp_tn['gene_1'] == genes[1]) &
                   (pair_tp_tn['gene_2'] == genes[0]))
            ]
        return [title, data_row['TP'], data_row['TN']]

    coords_df = pd.DataFrame()
    data = list(map(get_data, list(plot_genes)))
    coords_df[['title', 'TP', 'TN']] = pd.DataFrame(
        data, columns=['title', 'TP', 'TN']
    )

    plt.scatter(coords_df['TP'], coords_df['TN'], s=3)
    for index, row in coords_df.iterrows():
        plt.annotate(row['title'], (row['TP'] + PADDING, row['TN'] + PADDING))

    fig.savefig(path)
    plt.close(fig)
