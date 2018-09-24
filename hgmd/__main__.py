import os
import argparse

from . import hgmd as md
import sys


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
    # TODO: get X and L as CLI args

    # TODO: get command line input instead of assigning directly
    parser = argparse.ArgumentParser(
        description=("Hypergeometric marker detection. Finds markers "
                     "identifying a cluster.")
    )
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
        'output_path', type=str,
        help="the output directory where output files should go"
    )
    parser.add_argument(
        '-X', nargs='?', default=None,
        help="X argument for XL-mHG"
    )
    parser.add_argument(
        '-L', nargs='?', default=None,
        help="L argument for XL-mHG"
    )
    args = parser.parse_args()
    output_path = args.output_path
    X = args.X
    L = args.L
    marker_file = args.marker
    tsne_file = args.tsne
    cluster_file = args.cluster
    min_exp_ratio = 0.4
    plot_pages = 15
    plot_genes = 15

    if X is not None:
        X = int(X)
        print("Set X to " + str(X) + ".")
    if L is not None:
        L = int(L)
        print("Set L to " + str(L) + ".")
    print("Reading data...")
    cell_data = md.get_cell_data(
        marker_path=(marker_file),
        tsne_path=(tsne_file),
        cluster_path=(cluster_file)
    )

    # Enumerate clusters and process each individually in its own folder.
    # pair_data also contains singleton data, but singleton is just
    # singleton.
    clusters = cell_data['cluster'].unique()
    clusters.sort()
    for cluster in clusters:
        print("Processing cluster " + str(cluster) + "...")
        cluster_path = output_path + "/cluster_" + str(cluster) + "/"
        os.makedirs(cluster_path, exist_ok=True)
        print("Testing singletons...")
        singleton_data = md.singleton_test(cell_data, cluster, X, L)
        print("Testing pairs...")
        pair_data = md.pair_test(
            cell_data, singleton_data, cluster, L, min_exp_ratio
        )
        print("Calculating true positive/negative rates...")
        singleton_data, pair_data = md.find_TP_TN(
            cell_data, singleton_data, pair_data, cluster
        )
        print("Calculating weighted TP/TN rates...")
        singleton_data, pair_data = md.find_weighted_TP_TN(
            cell_data, singleton_data, pair_data, cluster
        )
        print("Saving to CSV...")
        singleton_data.to_csv(cluster_path + "singleton_data.csv")
        pair_data.to_csv(cluster_path + "pair_data.csv")
        print("Done.")
        print("Plotting true positive/negative rates...")
        md.make_TP_TN_plots(
            cell_data, singleton_data, pair_data, plot_genes,
            pair_path=(cluster_path + "TP_TN_plot.pdf"),
            singleton_path=(cluster_path + "singleton_TP_TN_plot.pdf")
        )
        print("Done.")
        print("Plotting discrete expression...")
        md.make_discrete_plots(
            cell_data, singleton_data, pair_data, plot_pages,
            path=(cluster_path + "discrete_plots.pdf"),
        )
        print("Done.")
        print("Plotting continuous expresssion...")
        md.make_combined_plots(
            cell_data, singleton_data, pair_data, plot_pages,
            pair_path=(cluster_path + "combined_plot.pdf"),
            singleton_path=(cluster_path + "singleton_combined_plot.pdf")
        )
        print("Done.")

    print("All set!!! Enjoy your PDFs and CSVs.")


if __name__ == '__main__':
    main()
