import csv
import os
import pandas as pd
import numpy as np
DPATH = '../data'
Drug_feature_file = '%s/CCLE/omics_data' % DPATH
Genomic_mutation_file_new = '%s/CCLE/CCLE_mutations.csv' % DPATH
Methylation_file_new = '%s/CCLE/CCLE_RRBS_TSS_1kb_20180614.txt' % DPATH
cell_line_name_to_DepMap_ID_file='%s/CCLE/sample_info.csv' % DPATH

cell_line_name_to_DepMap_ID={}
#Conversion dictionary
all_omic_info={}
#The total data includes four items

for each in os.listdir(Drug_feature_file):
    cellline_line_id=(each.split(".")[0])
    with open (Drug_feature_file+"/"+each,'r') as f:
        reader=csv.reader(f)
        next(reader, None)
        ACH_info={}
        for i in (reader):
            genename=i[0]
            genedata=i[1:]
            genedata.append(0)#0 here is prepared for mutation data
            assert  genename  not in ACH_info.keys()
            ACH_info[genename]=genedata
        all_omic_info[each.split(".")[0]]=ACH_info
#Save all existing information (copy num, expression) as a dictionary
#   all_omic_info[ACH_X]=ACH_info
#   ACH_info[gene_name]=[data1,data2]


with open (cell_line_name_to_DepMap_ID_file,'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for i in reader:
        cell_line_name_to_DepMap_ID[i[2]]=i[0]
sample=list(all_omic_info.keys())[0]
all_gene_list=[gene for gene in list(all_omic_info[sample].keys())]
#Store all required gene sequences

data_methylation = pd.read_csv(Methylation_file_new,sep='\t',na_values="     NA")
for gene_need  in data_methylation["gene"]:
    if gene_need in all_gene_list:
        data_methylation_in_this_gene=(data_methylation.loc[data_methylation["gene"]==gene_need])
        #Take the corresponding gene column. If the gene is the required gene, take out the data in this column
        for CCLE_name in data_methylation_in_this_gene.columns :
            if CCLE_name in list(cell_line_name_to_DepMap_ID.keys()):
                DepMap_ID = cell_line_name_to_DepMap_ID[CCLE_name]
                # If the corresponding cellline can be converted to ACH, then convert
                if  DepMap_ID in list(all_omic_info.keys()):
                    data_gene_cellline=(data_methylation_in_this_gene[CCLE_name].mean())
                    #NA is not selected here, that is, there is no empty data
                    if len( all_omic_info[DepMap_ID][gene_need])==3:
                        all_omic_info[DepMap_ID][gene_need].append(data_gene_cellline)
                    #If cellline is the selected cellline, find the corresponding data and add it


median_dic={}
for CCLE_name in data_methylation.columns:
    if CCLE_name in list(cell_line_name_to_DepMap_ID.keys()):
        median=data_methylation[CCLE_name].fillna(value=0).median()
        median_dic[cell_line_name_to_DepMap_ID[CCLE_name]]=median
#the median value dict
for cell_name in list(all_omic_info.keys()):
    for i in list(all_omic_info[cell_name].keys()):
        if len(all_omic_info[cell_name][i])==3:
            all_omic_info[cell_name][i].append(median_dic[cell_name])
#For genes that do not exist, the median value

with open (Genomic_mutation_file_new,'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        gene_name=line[0]
        cellline_id=line[15]
        if cellline_id in all_omic_info.keys():
            if gene_name in all_omic_info[cellline_id].keys():
                all_omic_info[cellline_id][gene_name][2]+=1
#Take out the corresponding data. If the gene is mutated, add one to the value
print(all_omic_info)