# file used to analyze the entailment and specificity score
import matplotlib.pyplot as plt
import numpy as np
from tree import Tree
from pathlib import Path


def get_avg_specificity(tree):
    avg_specificity = 0
    length = 0
    for level in tree.levels:
        for node in level.nodes:
            specificity = node.specificity
            if specificity is not None:
                avg_specificity+=specificity
                length+=1
    return avg_specificity/length
    

def get_avg_entail(tree):
    avg_entail = 0
    length = 0
    for level in tree.levels:
        for node in level.nodes:
            entail = node.entail_2
            if entail is not None:
                avg_entail+=entail
                length+=1
    return avg_entail/length
    


def plot_specificity_vs_entailment(*trees_lists, save_name):
    # gather avg specificity for each tree
    avg_specificity_list = [[get_avg_specificity(tree) for tree in trees] for trees in trees_lists]
    print(avg_specificity_list)
    # gather avg entailment score for each tree
    avg_entail_list = [[get_avg_entail(tree) for tree in trees] for trees in trees_lists]
    print(avg_entail_list)
    fig,ax = plt.subplots()
    haiku_plot = ax.scatter(avg_specificity_list[0],avg_entail_list[0])
    sonnet_plot = ax.scatter(avg_specificity_list[1],avg_entail_list[1])
    ax.set_xlabel("Specificity Averages")
    ax.set_ylabel("Entail-2 Averages")
    ax.legend([haiku_plot, sonnet_plot], ["Haiku", "Sonnet"])
    
    plt.text(avg_specificity_list[1][0], avg_entail_list[1][0]-0.01, "w specificity 1", fontsize=9)
    # plt.text(avg_specificity_list[1][1], avg_entail_list[1][1]-0.01, "w specificity 2", fontsize=9)
    plt.text(avg_specificity_list[1][1], avg_entail_list[1][1]-0.01, "wo specificity 1", fontsize=9)
    # plt.text(avg_specificity_list[1][3], avg_entail_list[1][3]-0.01, "wo specificity 2", fontsize=9)

    
    plt.text(avg_specificity_list[0][0], avg_entail_list[0][0]-0.01, "w specificity 1", fontsize=9)
    # plt.text(avg_specificity_list[0][1], avg_entail_list[0][1]-0.01, "w specificity 2", fontsize=9)
    plt.text(avg_specificity_list[0][1], avg_entail_list[0][1]-0.01, "w/o specificity 1", fontsize=9)
    # plt.text(avg_specificity_list[0][3], avg_entail_list[0][3]-0.01, "w/o specificity 2", fontsize=9)
    plt.savefig(save_name)
    
    # plt.show()
    # pass    

if __name__ == "__main__":
    
    # pipline for the plotting: list of json files --> tree --> plot

    # load Haiku without specificity
    # print("loading haiku w specificity")
    haiku_w_specificity = Tree.load("uniform_64_haiku_sumv5_4o-mini_evalv1.json")
    # # load Haiku with specificity
    # print("loading haiku wo specificity")
    haiku_wo_specificity = Tree.load("uniform_64_haiku_sumv6_4o-mini_evalv1.json")
    
    # print(f"Average specificity for haiku and without specificity: {get_avg_specificity(haiku_wo_specificity)}")
    # print(f"Average specificity for haiku and with specificity: {get_avg_specificity(haiku_w_specificity)}")

    # print(f"Average entail-1 for haiku and without specificity: {get_avg_entail(haiku_wo_specificity)}")
    # print(f"Average entail-1 for haiku and with specificity: {get_avg_entail(haiku_w_specificity)}")

    # # load Sonnet without specificity
    # # load sonnet with specificity
    # print("loading sonnet w specificity")
    sonnet_w_specificity = Tree.load("uniform_64_sonnet_sumv5_4o-mini_evalv1.json")
    # # load Haiku with specificity
    # print("loading haiku wo specificity")
    sonnet_wo_specificity = Tree.load("uniform_64_sonnet_sumv6_4o-mini_evalv1.json")
    
    # haiku_w_specificity2 = Tree.load("trees/10NNofAevalpaircodesumv5-2.json")
    # haiku_wo_specificity2 = Tree.load("trees/10NNofAevalpaircodesumv6-2.json")
    
    # sonnet_w_specificity2 = Tree.load("trees/10NNofAevalpaircodesumv5sonnet-2.json")
    # sonnet_wo_specificity2 = Tree.load("trees/10NNofAevalpaircodesumv6sonnet-2.json")
    
    # sonnet_64_wo_specificity = Tree.load('uniform_64_test.json')
    # print(f"Average specificity for sonnet and without specificity: {get_avg_specificity(sonnet_wo_specificity)}")
    # print(f"Average specificity for sonnet and with specificity: {get_avg_specificity(sonnet_w_specificity)}")

    # print(f"Average entail-1 for sonnet and without specificity: {get_avg_entail(sonnet_wo_specificity)}")
    # print(f"Average entail-1 for sonnet and with specificity: {get_avg_entail(sonnet_w_specificity)}")

    plot_specificity_vs_entailment([haiku_w_specificity, haiku_wo_specificity],[sonnet_w_specificity, sonnet_wo_specificity], save_name='specificity_vs_entail_64_v1.png')
                                   
    
    # feed trees to plotter
    # pass