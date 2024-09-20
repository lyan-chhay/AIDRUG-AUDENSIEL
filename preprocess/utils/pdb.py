import requests


def pdb2uniprot(pdb_id):
    url = 'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}'
    response = requests.get(url.format(pdb_id))
    if response.status_code == 200:
        return  list(response.json()[pdb_id]['UniProt'].keys())[0]
    else:
        return None
    
def uniprot2pdb(uniprot_id):
    url = 'https://www.ebi.ac.uk/pdbe/api/mappings/{}'
    response = requests.get(url.format(uniprot_id))
    if response.status_code == 200:
        return  list(response.json()[uniprot_id]['PDB'].keys())
    else:
        return None