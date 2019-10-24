# Standard libraries
import numpy as np
import pandas as pd

# Various
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def remove_corrupted_images(df):
    bad_image_ids = set(1,2,3)
    if df.ID in bad_image_ids:
        filter = 0
    else:
        filter = 1
    return filter


    return 1
def clean_data_csv(data_path = "../data/", csv_name = 'stage_1_train.csv',shuffle_data = True):

    data_raw = pd.read_csv(data_path + csv_name)
    data_raw['filename'] = data_raw['ID'].apply(lambda x: "ID_" + x.split('_')[1] + ".dcm")
    data_raw['type'] = data_raw['ID'].apply(lambda x: x.split('_')[2])

    data_pivot = data_raw[['Label', 'filename', 'type']].drop_duplicates().pivot(
        index='filename', columns='type', values='Label').reset_index()
    
    if shuffle_data:
        data = shuffle(data_pivot)
    else:
        data = data_pivot
    data.reset_index(drop=True,inplace=True)
    data = pd.DataFrame(data,columns = list(data.columns))
        
    return data

def balanced_images_all_classes(image_list,n=2500,replace=False,random_state = 12345):
    image_subset = [
        image_list[image_list[category] ==1 ].sample(
            n=n,
            replace = replace, 
            random_state = random_state)
        for category
        in ['epidural',
            'intraparenchymal',
            'intraventricular',
            'subarachnoid',
            'subdural']
    ]
    image_subset.append(
        image_list[image_list['any'] == 0 ].sample(
            n=n*5,
            replace = replace, 
            random_state = random_state)
    )
   
    image_subset_combined = pd.concat(image_subset).drop_duplicates()
    image_subset_combined = shuffle(image_subset_combined,random_state = random_state)
    image_subset_combined.reset_index(drop=True, inplace = True)
    return image_subset_combined


def balanced_images_binary(image_list,n=2500,replace=False,random_state = 12345):
    image_subset = [image_list[image_list['any'] ==1 ]]
    image_subset.append(
        image_list[image_list['any'] == 0 ].sample(
            n=len(image_subset[0]),
            replace = replace, 
            random_state = random_state)
    )
    
    image_subset_combined = pd.concat(image_subset).drop_duplicates()
    image_subset_combined = shuffle(image_subset_combined,random_state = random_state)
    image_subset_combined.reset_index(drop=True,inplace=True)
    return image_subset_combined

def corrupted_images():
    """IDs of corrupted images"""
    return ['ID_cec3997fa',
     'ID_f242fed92',
     'ID_ca9462f49',
     'ID_22069463a',
     'ID_d1afb9750',
     'ID_cb970c6dc',
     'ID_53c71fb9d',
     'ID_18aac96c0',
     'ID_10fe2031e',
     'ID_c2738e8b1',
     'ID_d3fd5220e',
     'ID_3e60e696d',
     'ID_451f60160',
     'ID_807b56a94',
     'ID_ae1689e1b',
     'ID_4e61fb0b2',
     'ID_f188940f9',
     'ID_b194d2a23',
     'ID_046ba342c',
     'ID_ca4a832a1',
     'ID_a7e689932',
     'ID_85900eb84',
     'ID_8c5fc9e44',
     'ID_680b2194c',
     'ID_1e633cf27',
     'ID_97cd49666',
     'ID_b055aafa9',
     'ID_a356248db',
     'ID_d77fa1286',
     'ID_676b0cb59',
     'ID_8fc348d44',
     'ID_950a06268',
     'ID_9ece1bb21',
     'ID_ced5fabca',
     'ID_631f0b556',
     'ID_db48a633d',
     'ID_82ec3b736',
     'ID_a3feeadf4',
     'ID_3bc141392',
     'ID_b12bb2b16',
     'ID_12e3b6923',
     'ID_94463e98f',
     'ID_c6f2d84be',
     'ID_6cc19ac41',
     'ID_15b3ba199',
     'ID_6c19c9f7b',
     'ID_0b0e59911',
     'ID_b77ba3355',
     'ID_72dce7784',
     'ID_c64131283',
     'ID_0c4987103',
     'ID_64b44f180',
     'ID_b76de950b',
     'ID_3eb407dd8',
     'ID_c1ff9eb46',
     'ID_dfa4e344a',
     'ID_c35d5c858',
     'ID_b494c2115',
     'ID_21053fe7e',
     'ID_ea0ddbaf9',
     'ID_91b9ce430',
     'ID_bd4f3f06f',
     'ID_cef2af72d',
     'ID_c6463f07d',
     'ID_6fbc30b5d',
     'ID_ac1d14c29',
     'ID_3274f5977',
     'ID_66131f4c9',
     'ID_798d956d0',
     'ID_c65ca5466',
     'ID_4e0bdd2ba',
     'ID_f03370d7c',
     'ID_ff012ee5b',
     'ID_767c42624',
     'ID_ba7080372',
     'ID_842e85173',
     'ID_4f0317d23',
     'ID_73dee8958',
     'ID_7714ead69',
     'ID_1bc5771a7',
     'ID_3ba8a116c',
     'ID_66accd2e4',
     'ID_55f7bbbf2',
     'ID_6a939bc17',
     'ID_942e2f95b',
     'ID_d6435f3bf',
     'ID_fe7327fab',
     'ID_56ecdf5c1',
     'ID_fd5c41761',
     'ID_a9ab8569f',
     'ID_3f422852d',
     'ID_2b3671dd9',
     'ID_17103c79e',
     'ID_69974dd3e',
     'ID_19f266244',
     'ID_5dbe845c1',
     'ID_11c4f9f91',
     'ID_04280250b',
     'ID_7e870621c',
     'ID_8d0ca7742',
     'ID_aef6c6df9',
     'ID_567a36143',
     'ID_6cb797177',
     'ID_60a1f0e24',
     'ID_b1cea5abb',
     'ID_317330708',
     'ID_5bf2ca43f',
     'ID_6d7a27643',
     'ID_6b15a7649',
     'ID_f23f8e617',
     'ID_079945c27',
     'ID_081f4d071',
     'ID_10f34fb10',
     'ID_8144c7120',
     'ID_6dcedd2e1',
     'ID_d3b76ef6e',
     'ID_25de55880',
     'ID_b76b13444',
     'ID_184c541fa',
     'ID_19306ecc5',
     'ID_362423b57',
     'ID_23d0b13b7',
     'ID_97e5a203e',
     'ID_a2e178cc7',
     'ID_5ab140176',
     'ID_e4b636907',
     'ID_b966185b8',
     'ID_0de0ab1d8',
     'ID_d1b2d9ad0',
     'ID_d4ea87a35',
     'ID_c60e34466',
     'ID_f698edc00',
     'ID_abcd58e88',
     'ID_dabc2a818',
     'ID_d7777de78',
     'ID_845f922f4',
     'ID_cf4d76860',
     'ID_5ffae2e26',
     'ID_ab474037b',
     'ID_f1fe5334e',
     'ID_8dc299456',
     'ID_8caa68ebd',
     'ID_a40f9b2de',
     'ID_1bb3b44c7',
     'ID_4c9fb82af',
     'ID_155249efa',
     'ID_4e14d0fe8',
     'ID_0f8aa5749',
     'ID_9a36e4b0e',
     'ID_d1a1c9a6c',
     'ID_3d5d23058',
     'ID_c6bbec638',
     'ID_b19f52c76',
     'ID_53f460f86',
     'ID_882cd57de',
     'ID_88b0d8b4f',
     'ID_176e4f16d',
     'ID_c11582dc9',
     'ID_898ff55b6',
     'ID_b4adf8739',
     'ID_de10fdac2',
     'ID_80a2dbc4a',
     'ID_f0d55b727',
     'ID_7917d368d',
     'ID_a9e98ab5e',
     'ID_9da128021',
     'ID_9cdc7295b',
     'ID_f4c2157d8',
     'ID_ae691dd29',
     'ID_2ac7f01ed',
     'ID_dd083e12a',
     'ID_75cbdae68',
     'ID_c1a3f037f',
     'ID_1291d1943',
     'ID_1690a6499',
     'ID_3c8b72361',
     'ID_09aeb0bbd',
     'ID_28d6a694f',
     'ID_a1bb9bc26',
     'ID_c45659d3d',
     'ID_0144e4030',
     'ID_155b9c546',
     'ID_76f88846f',
     'ID_c964e4096',
     'ID_61d2718d2',
     'ID_a23a8193f',
     'ID_d7229490a',
     'ID_be3fb6c17',
     'ID_36ab2e72a',
     'ID_dd3b5bf4e',
     'ID_f145c3cf4',
     'ID_a2f9ba4bf',
     'ID_ac39010dc',
     'ID_ac47ba810',
     'ID_0b2ec2d3f',
     'ID_8f5d4b696',
     'ID_57d6a6455',
     'ID_0e1861e6d',
     'ID_9b297fa83',
     'ID_9dad2eb09',
     'ID_9b68c3f5f',
     'ID_2fd4dda7c',
     'ID_bb2a4a01c',
     'ID_ff9674e53',
     'ID_f4891876d',
     'ID_0603b315e',
     'ID_0e9ac1c5f',
     'ID_2e690fe7c',
     'ID_cbbb50e6d',
     'ID_44d57858e',
     'ID_8fd6d5047',
     'ID_91c508c7a',
     'ID_12a0d6d34',
     'ID_c037d5727',
     'ID_6b1a86148',
     'ID_3d7a23dbb',
     'ID_3dcbd1b5e',
     'ID_6508563e0',
     'ID_191369dca',
     'ID_3cb1b59bc',
     'ID_9bc2b62cc',
     'ID_830f46cad',
     'ID_f22730d7b',
     'ID_c07d2cb73',
     'ID_37c495912',
     'ID_cade293be',
     'ID_68e45bca7',
     'ID_8a35660d5',
     'ID_c4575f13b',
     'ID_ae7b11865',
     'ID_ae7020fd1',
     'ID_49ecc6164',
     'ID_b9938c32c',
     'ID_a880e377e',
     'ID_3e31d57d0',
     'ID_75e3f7e5a',
     'ID_bc97a5f4f',
     'ID_c51cbe76b',
     'ID_dfaa49f5c',
     'ID_6f92e4481',
     'ID_61c646098',
     'ID_7e756c43b',
     'ID_def2a0e9f',
     'ID_7c08b7fb7',
     'ID_fdbfb2c17',
     'ID_75d691728',
     'ID_142f85eb8',
     'ID_394ffb5fd',
     'ID_21d4bd6f3',
     'ID_a432727fd',
     'ID_291edd834',
     'ID_038f966b9',
     'ID_28c4609b3',
     'ID_27757c171',
     'ID_a3128aa77',
     'ID_6431af929', # the corrupted one
     'ID_ea2861e9a',
     'ID_7940bb7d0',
     'ID_b8665a653',
     'ID_8fde47d9f',
     'ID_403b4fc67',
     'ID_d9840380c',
     'ID_7607dbd07',
     'ID_445a92ac2',
     'ID_8756b0c04',
     'ID_9a3bba619',
     'ID_7a02fdbea',
     'ID_d0c52575a',
     'ID_985fb5e49',
     'ID_af129aa8e']