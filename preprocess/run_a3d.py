
# pip install tqdm
# !pip install 

import os
from tqdm import tqdm
import pandas as pd
from pprint import pprint as pp
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess 
import os
import shutil
import pandas as pd
# from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord

######################################################################


# def dataframe_to_fasta(df, id_col, seq_col, desc_col, output_file):
#     """
#     Converts a pandas DataFrame to a FASTA file.
    
#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the sequence data.
#     id_col (str): The column name for sequence IDs.
#     seq_col (str): The column name for sequences.
#     desc_col (str): The column name for descriptions.
#     output_file (str): The output FASTA file path.
#     """
#     # Convert DataFrame to a list of SeqRecord objects
#     records = []
#     for index, row in df.iterrows():
#         seq_record = SeqRecord(Seq(row[seq_col]), id=row[id_col], description=row[desc_col])
#         records.append(seq_record)
    
#     # Write the SeqRecord objects to a FASTA file
#     with open(output_file, 'w') as output_handle:
#         SeqIO.write(records, output_handle, 'fasta')
    
    
# def fasta_to_dataframe(fasta_file):
#     """
#     Converts a FASTA file to a pandas DataFrame.
    
#     Parameters:
#     fasta_file (str): The path to the FASTA file.
    
#     Returns:
#     pd.DataFrame: A DataFrame with columns 'ID', 'Sequence', and 'Description'.
#     """
#     # Read the FASTA file
#     records = list(SeqIO.parse(fasta_file, 'fasta'))
    
#     # Extract data and create DataFrame
#     data = {
#         'ID': [record.id for record in records],
#         'Sequence': [str(record.seq) for record in records],
#         'Description': [' '.join(record.description.split()[1:]) for record in records]
#     }
#     df = pd.DataFrame(data)
    
#     return df



def copy_files(pdb,dir_from='tmp/', dir_to='data/'):
    files = {
        dir_from+"/%s/A3D.csv" % pdb: dir_to+"/score/%s.csv" % pdb,
        dir_from+"/%s/output.pdb" % pdb: dir_to+"/pdb/%s.pdb" % pdb
    }
    
    # Create necessary directories
    for dest in files.values():
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    
    # Copy files with error handling
    for src, dest in files.items():
        try:
            shutil.copy(src, dest)
            # print("Copied %s to %s" % (src, dest))

        except IOError as e:
            print("%s: Error copying %s to %s: %s" % (pdb, src, dest, e))
            return pdb
        # If all files are copied successfully, remove the tmp directory
    
    try:
        shutil.rmtree("tmp/%s" % pdb)
        # print("Removed directory tmp/%s" % pdb)
    except OSError as e:
        print("%s: Error removing directory tmp/%s: %s" % (pdb, pdb, e))



def run_aggrescan(pdb_id, stabalize=True, dynamic=False, wdir ='tmp/'):
    # script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    # foldx_path = os.path.join(script_dir, _FOLDX_PATH)
    # print(foldx_path)

    command = ["aggrescan", "-i", pdb_id, "-w", wdir+ "{}".format(pdb_id)]
    if stabalize:
        command.append("-f")
    if dynamic:
        command.append("-d")
    # if chain_id is not None:
    #     command.append("-C {}".format(chain_id))

    # print("Running command: {}".format(' '.join(command)))
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("pdb {} : Command succeeded".format(pdb_id))
            copy_files(pdb_id,'tmp/','../data/')
            return pdb_id, True
        else:
            # print("pdb {} : Command failed ".format(pdb_id))
            try:
                shutil.rmtree("tmp/%s" % pdb_id)
                # print("Removed directory tmp/%s" % pdb)
            
            except OSError as e:
                print("%s: Error removing directory tmp/%s: %s" % (pdb_id, pdb_id, e))
            
            return pdb_id, False

    except Exception as e:
        print("pdb {} : Command failed with exception: {}".format(pdb_id, e))
        process.kill()
        return pdb_id, False
    

def run_aggrescan_all(pdbs,stabalize=True, dynamic=False, max_workers=16):

    # Directory to store temporary files
    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # if chain_ids is None:
    #     chain_ids = [None]*len(pdbs)

    # failed_pdbs = []
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # Submit tasks for each PDB ID
    #     results = [executor.submit(run_aggrescan, pdb_id, chain_id, stabalize, dynamic, tmp_dir)
    #                for pdb_id, chain_id in zip(pdbs, chain_ids)]

    #     # Wait for all tasks to complete
    #     print("Running Aggrescan for PDBs:")
    #     for future in tqdm(as_completed(results), total=len(pdbs)):
    #         pdb_id, succeed = future.result()
    #         # print(pdb_id, succeed)
            
    #         if succeed == False:
    #         # if pdb_id is not None and succeed =='False' and pdb_id not in failed_pdbs:
    #             failed_pdbs.append(pdb_id)
            
    # return failed_pdbs
    failed_pdbs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each PDB ID
        results = [executor.submit(lambda a: run_aggrescan(*a), args)
            for args in zip(pdbs, [stabalize]*len(pdbs))]

        # Wait for all tasks to complete
        print("Running Aggrescan for PDBs:")
        for future in tqdm(as_completed(results), total=len(pdbs)):
            pdb_id, succeed = future.result()
            # print(pdb_id, succeed)
            
            if succeed == False:
            # if pdb_id is not None and succeed =='False' and pdb_id not in failed_pdbs:
                failed_pdbs.append(pdb_id)
            
    return failed_pdbs


def read_txt_to_dataframe(file_path):
    # Read the file into a DataFrame, using the first line as header
    df = pd.read_csv(file_path, sep=r'\s+')
    
    # Replace 'NA' with NaN for proper handling
    df.replace('NA', pd.np.nan, inplace=True)

    # Split the PDBchain column into PDB_id and chain
    df[['PDB', 'chain']] = df['PDBchain'].str.extract(r'(\w{4})(.*)')

    # Drop weird row
    rows_to_drop = df[df['chain'].str.len() > 1].index
    df.drop(rows_to_drop, inplace=True)
    # df.drop(columns=['PDBchain'], inplace=True)
    
    return df


def find_dirs_in_tmp():
    tmp_path = 'tmp/'
    if not os.path.exists(tmp_path):
        print("The directory 'tmp/' does not exist.")
        return []
    
    all_entries = os.listdir(tmp_path)
    dirs = [entry for entry in all_entries if os.path.isdir(os.path.join(tmp_path, entry))]
    
    return dirs
##########################################################################################

pdb_chain_ids = []


## torun.txt contain all pdb to run aggrescan3d

with open("torun.txt", "r") as f:
    for line in f:
        pdb_chain_ids.append(line.strip())

print(pdb_chain_ids)
pdb_ids = [id[:4] for id in pdb_chain_ids]
chain_ids = [id[4] for id in pdb_chain_ids]


print(pdb_ids)
print(len(pdb_ids))



# ####################
# #### run aggrescan3D : take 38minute to finish to around 2700 pdb
# ####################


# run_aggrescan('1XKZ',False)
failed_pdbs = run_aggrescan_all(pdb_ids,stabalize=False, dynamic=False, max_workers=20)


with open("failed_pdbs.txt", "w") as f:
    for pdb_id in failed_pdbs:
        f.write(pdb_id + "\n")



# # ####################
# # #### resume run aggrescan3D : resume to run aggrescan
# # ####################
# def get_pdb_ids(directory):
#     pdb_ids = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.pdb'):
#             pdb_id = filename[:-4]  # Remove the '.pdb' extension
#             pdb_ids.append(pdb_id)
#     return pdb_ids

# def get_pdb_ids_csv(directory):
#     pdb_ids = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):
#             pdb_id = filename[:-4]  # Remove the '.pdb' extension
#             pdb_ids.append(pdb_id)
#     return pdb_ids

# def intersect_lists(list1, list2):
#     set1 = set(list1)
#     set2 = set(list2)
#     return list(set1 & set2)

# def not_intersect_lists(list1, list2):
#     set1 = set(list1)
#     set2 = set(list2)
#     return list(set2 - set1)


# done_pdbs = get_pdb_ids_csv("/users/eleves-b/2023/ly-an.chhay/main/data/score/")
# undone_pdbs = not_intersect_lists(done_pdbs,pdb_ids)

# print(len(done_pdbs))
# print(len(undone_pdbs))

# # failed_pdbs = run_aggrescan_all(undone_pdbs,chain_ids,stabalize=False, dynamic=False, max_workers=1)
# failed_pdbs = run_aggrescan_all(undone_pdbs,None,stabalize=False, dynamic=False, max_workers=10)

# # # failed_pdbs = run_aggrescan_all(undone_pdbs[100:110],stabalize=False, dynamic=False, max_workers=16)
# # # failed_pdbs = run_aggrescan_all(undone_pdbs[:50],stabalize=True, dynamic=False, max_workers=16)



# ########################################
# with open("failed_pdbs.txt", "w") as f:
#     for pdb_id in failed_pdbs:
#         f.write(pdb_id + "\n")


## print(len(failed_pdbs), 'failed among ', len(undone_pdbs))


# ############################################################
# # save just the PDB without problem with aggrescan
# ###########################################################

# # filtered_df30 = df30[df30['PDB'].isin(done_pdbs)]
# # filtered_df30.to_csv("/users/eleves-b/2023/ly-an.chhay/stage/pisces/data30.csv", index=False)