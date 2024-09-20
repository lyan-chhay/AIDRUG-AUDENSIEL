import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import wget
import time


def df2fasta(df, fasta_file):
    # Extract column names except 'seq'
    description_columns = [col for col in df.columns if col != 'seq']

    directory = os.path.dirname(fasta_file)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        print(f"Error: {directory} - {e.strerror}")
        
    with open(fasta_file, 'w') as f:
        for index, row in df.iterrows():
            # Create description by joining column names and values
            description = '|'.join([f"{col}:{str(row[col]).replace('|', '')}" for col in description_columns])
            fasta_entry = f">{description}\n{row['seq']}\n"
            f.write(fasta_entry)


def fasta2df(fasta_file_path):
    # Initialize empty lists to store data
    headers = []
    sequences = []
    header_fields = set()
    
    # # Split the input string into lines
    # lines = fasta_str.strip().split('\n')
    
    # Iterate over lines to parse headers and sequences
    with open(fasta_file_path, 'r') as fasta_file:
        current_sequence = ''
        for line in fasta_file:
            if line.startswith('>'):
                # Header line
                header = line[1:]
                headers.append(header)
                fields = header.split('|')
                header_fields.update({field.split(':')[0] for field in fields})
                if current_sequence:  # If there's a sequence, append it before starting a new one
                    sequences.append(current_sequence)
                    current_sequence = ''
            else:
                # Sequence line
                current_sequence += line.strip()
    
    # Append the last sequence after the loop
    sequences.append(current_sequence)
    
    # Create a dictionary for columns
    columns = {key: [] for key in header_fields}
    columns['seq'] = []

    print(columns.keys())
    
    # Parse headers and sequences to populate columns
    for header, sequence in zip(headers, sequences):
        fields = header.split('|')
        header_dict = {field.split(':')[0]: field.split(':')[1] for field in fields}
        for key in columns.keys()- {'seq'}:
            if key in header_dict:
                columns[key].append(header_dict[key])
            else:
                columns[key].append(None)
        columns['seq'].append(sequence)

    print(columns)
    
    # Create a DataFrame
    df = pd.DataFrame(columns)
    
    return df


def download_file_wget(url, output_path, verbose=False):
    """
    Download a file from the given URL using wget and save it to the specified output path.
    
    Args:
        url (str): The URL of the file to download.
        output_path (str): The path where the downloaded file will be saved.
        verbose (bool): Whether to print status messages. Default is True.
        max_retries (int): Maximum number of times to retry downloading on timeout. Default is 3.
        
    Returns:
        bool: True if the download was successful, False otherwise.
    """

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        if verbose:
            print("File already exists at:", output_path)
        return True
    retries = 3

    while retries > 0:
        try:
            if verbose:
                print("Downloading file from:", url)
            wget.download(url, out=output_path)
            return True
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            elif isinstance(e, (TimeoutError, ConnectionError)):
                if verbose:
                    print("Timeout error occurred. Retrying...")
                retries -= 1
                time.sleep(1)  # Wait for a second before retrying
            else:
                if verbose:
                    print("Error:", str(e))
                return False
    return False



# def extract_and_remove_tar(tar_path):
#     """
#     Extracts a .tar file from the specified path and removes the extracted file.

#     Parameters:
#         tar_path (str): The path to the .tar file.
#     """
#     try:
#         # Extract the .tar file
#         with tarfile.open(tar_path, 'r') as tar:
#             tar.extractall(path=os.path.dirname(tar_path))

#         # Remove the extracted file
#         os.remove(tar_path)
        
#         print(f"Successfully extracted and removed {tar_path}")
#         return True
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return False


def download_aggrescan_score_files(uniprots , job_ids, 
                        score_file_path= None,
                        max_workers=16):
    """
    This function is used to download the aggrescan3d score files of protein from the Aggrescan3D database 
    based on the job_id from the summary file.
    """
    #create directories if they don't exist
    if score_file_path is None:
        score_file_path = os.path.join(os.getcwd(), 'data/score')
    os.makedirs(score_file_path, exist_ok=True)

    warning_uniprots = []

    score_file_paths = [
        os.path.join(score_file_path, uniprot + '.csv') for uniprot in uniprots
    ]

    template_url = 'https://biocomp.chem.uw.edu.pl/A3D2/compute_static/{}/A3D.csv'
    download_urls = [template_url.format(job_id) for job_id in job_ids]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_uniprot = {
            executor.submit(download_file_wget, url, path, False): uniprot 
            for url, path, uniprot in zip(download_urls, score_file_paths, uniprots)
        }
        print('Downloading aggrescan score files to {}...'.format(score_file_path))
        for future in tqdm(as_completed(future_to_uniprot), total=len(download_urls)):
            uniprot = future_to_uniprot[future]
            try:
                success = future.result()
                if not success:
                    warning_uniprots.append(uniprot)
            except Exception as e:
                warning_uniprots.append(uniprot)
                print(f"Error downloading file for {uniprot}: {e}")

    return warning_uniprots


def download_aggrescan_pdb_files(uniprots, job_ids, pdb_file_path=None, max_workers=16):
    """
    This function is used to download the Aggrescan3D PDB files of proteins from the Aggrescan3D database 
    based on the job_id from the summary file.
    """
    warning_uniprots = []

    # Create directories if they don't exist
    if pdb_file_path is None:
        pdb_file_path = os.path.join(os.getcwd(), 'data/pdb')
    os.makedirs(pdb_file_path, exist_ok=True)


    pdb_file_paths = [
        os.path.join(pdb_file_path, uniprot + '.pdb') for uniprot in uniprots
    ]

    template_url = 'https://biocomp.chem.uw.edu.pl/A3D2/compute_static/{}/output.pdb'
    download_urls = [template_url.format(job_id) for job_id in job_ids]


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_uniprot = {
            executor.submit(download_file_wget, url, path, False): uniprot 
            for url, path, uniprot in zip(download_urls, pdb_file_paths, uniprots)
        }
        print(f'Downloading Aggrescan PDB files to {pdb_file_path}...')
        for future in tqdm(as_completed(future_to_uniprot), total=len(download_urls)):
            uniprot = future_to_uniprot[future]
            try:
                success = future.result()
                if not success:
                    warning_uniprots.append(uniprot)
            except Exception as e:
                warning_uniprots.append(uniprot)
                print(f"Error downloading file for {uniprot}: {e}")

    return warning_uniprots








############################################################################################################

# def download_file_request(url, output_path, isprint=True):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an error for HTTP status codes indicating failure
        
#         with open(output_path, 'wb') as f:  # Use 'wb' mode for writing binary data
#             f.write(response.content)
#         if isprint:
#             print(f"Successfully downloaded {url} to {output_path}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading {url}: {e}")
