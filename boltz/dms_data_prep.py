import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import glob
import torch

def hgvs_pro_aminos_wt(p_string):
  '''take in an hgvs protein variant string and return the relevant WT aa'''
  aa_list = re.split(r'\d+',p_string)
  return aa_list[0][2:]

def hgvs_pro_aminos_alt(p_string):
  '''take in an hgvs protein variant string and return the relevant mutated aa'''
  aa_list = re.split(r'\d+',p_string)
  if aa_list[1] == '=': #replace the equal sign with the wt amino acid 
    aa_list = re.split(r'\d+',p_string)
    return aa_list[0][2:]
  return aa_list[1]

def hgvs_pro_aminos_pos(p_string):
  '''take in an hgvs protein variant string and return the relevant aa position in protein seq'''
  aa_pos = re.search(r'\d+',p_string)
  return p_string[aa_pos.start():aa_pos.end()]

def hgvs_mut_aa(p_string):
  c = re.search(r'([a-zA-Z]{3})(\d{1,})([a-zA-Z]{3}|=)', p_string)
  return amino_acid_dict_3_to_num[c.group(3)]

def hgvs_wt_aa(p_string):
  c = re.search(r'([a-zA-Z]{3})(\d{1,})([a-zA-Z]{3}|=)', p_string)
  return amino_acid_dict_3_to_num[c.group(1)]

def hgvs_num_aa(p_string):
  c = re.search(r'([a-zA-Z]{3})(\d{1,})([a-zA-Z]{3}|=)', p_string)
  return int(c.group(2))

def csv_to_maskedarray(df,score_name,protein_length):
  '''take in a pandas dataframe and place scores in a masked array'''

  scores_to_putinarray = df[[f'{score_name}','pos_aminoacid','alt_aminoacid']].to_numpy()
  score_array = np.ones((22,protein_length+1))*99999 # +1 because some maps have a termination mutations
  for score in scores_to_putinarray:
    col = int(score[1])-1
    row = int(score[2])-1
    score_array[row,col] = score[0]
  score_array_masked = np.ma.masked_values(score_array,99999)
  return score_array_masked

def csv_to_mask_overlay(df,protein_length):
  '''take in a pandas dataframe and return a mask array with 0 at the missing aa positions'''

  scores_to_putinarray = df[['score','pos_aminoacid','aaalt_num']].to_numpy()
  score_array = np.zeros((22,protein_length+1)) # +1 because some maps have a termination mutations
  for score in scores_to_putinarray:
    col = int(score[1]-1)
    row = int(score[2]-1)
    score_array[row,col] = 1

  return score_array


amino_acid_dict_3_to_1 = {'Ala': 'A',
 'Cys': 'C',
 'Asp': 'D',
 'Glu': 'E',
 'Phe': 'F',
 'Gly': 'G',
 'His': 'H',
 'Ile': 'I',
 'Lys': 'K',
 'Leu': 'L',
 'Met': 'M',
 'Asn': 'N',
 'Pro': 'P',
 'Gln': 'Q',
 'Arg': 'R',
 'Ser': 'S',
 'Thr': 'T',
 'Val': 'V',
 'Trp': 'W',
 'Tyr': 'Y'}

amino_acid_dict_3_to_num = {'Ala': 1,
 'Cys': 19,
 'Asp': 12,
 'Glu': 13,
 'Phe': 6,
 'Gly': 18,
 'His': 10,
 'Ile': 4,
 'Lys': 11,
 'Leu': 3,
 'Met': 5,
 'Asn': 16,
 'Pro': 20,
 'Gln': 17,
 'Arg': 9,
 'Ser': 14,
 'Thr': 15,
 'Val': 2,
 'Trp': 8,
 'Tyr': 7,
  '=':21,
  'Ter':22}

amino_acid_dict_1_to_num ={'A': 1,
 'C': 19,
 'D': 12,
 'E': 13,
 'F': 6,
 'G': 18,
 'H': 10,
 'I': 4,
 'K': 11,
 'L': 3,
 'M': 5,
 'N': 16,
 'P': 20,
 'Q': 17,
 'R': 9,
 'S': 14,
 'T': 15,
 'V': 2,
 'W': 8,
 'Y': 7,
  '=':21,
  'Ter':22}
path = '../DMS_DL_Impute/*.csv'
for filename in glob.glob(path):
    data = pd.read_csv(filename)
    print(filename)
    data['aa_pos'] = data['hgvs_pro'].map(hgvs_pro_aminos_pos)
    data['alt_aminoacid'] = data['hgvs_pro'].map(hgvs_mut_aa)
    data['wt_aminoacid'] = data['hgvs_pro'].map(hgvs_wt_aa)
    data['pos_aminoacid'] = data['hgvs_pro'].map(hgvs_num_aa)
    temp = csv_to_maskedarray(data,'score',data['pos_aminoacid'].max())
    filename = filename
    
    # Convert to torch tensor and save
    temp_tensor = torch.from_numpy(temp.filled(np.nan))  # fill masked values with nan
    torch.save(temp_tensor, filename.replace('.csv', '.pt'))