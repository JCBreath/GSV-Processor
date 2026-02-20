import os
import json
import pandas as pd
from supervision import Detections
import zipfile
import torch
from tqdm import tqdm

def create_state_bevs(state, gsv_proc_dir, gsv_bev_dir):
    gsv_proc_state_dir = f"{gsv_proc_dir}/{state}"
    gsv_bev_state_dir = f"{gsv_bev_dir}/{state}"
    groups = os.listdir(gsv_proc_state_dir)
    groups = sorted(groups, key=lambda x: int(x.split('_')[-2]))

    print(groups)
    os.makedirs(gsv_bev_dir, exist_ok=True)
    os.makedirs(f"{gsv_bev_dir}/{state}", exist_ok=True)

    completed_counties = []
    state_nsi_bev = {}

    if not os.path.exists(f"{gsv_bev_state_dir}/nsi.json"):
        raise ValueError(f"NSI JSON file not found at {gsv_bev_state_dir}/nsi.json. Please run with --nsi flag first to create the NSI JSON file.")

    with open(f"{gsv_bev_state_dir}/nsi.json", 'r') as f:
        print(f"Loading NSI data from {gsv_bev_state_dir}/nsi.json...")
        state_nsi_bev = json.load(f)
        state_metadata = {}

    for group in groups:
        counties = os.listdir(f"{gsv_proc_state_dir}/{group}")
        print(counties)
        for county in counties:
            print(f"Processing BEV for {state} {group} {county}...")
            
            gsv_proc_county_dir = f"{gsv_proc_state_dir}/{group}/{county}"
            gsv_proc_packed_cropped_dir = f"{gsv_proc_county_dir}/packed_cropped"
            metadata_log_dir = f"{gsv_proc_county_dir}/log_cropped_building"
            wallmat_log_dir = f"{gsv_proc_county_dir}/log_wall_material"
            condition_log_dir = f"{gsv_proc_county_dir}/log_building_condition"

            wallmat_sections = os.listdir(wallmat_log_dir)
            condition_sections = os.listdir(condition_log_dir)

            wallmat_sections = [s for s in wallmat_sections if s.endswith('.pth')]
            condition_sections = [s for s in condition_sections if s.endswith('.pth')]

            if len(wallmat_sections) == len(condition_sections):
                sections = wallmat_sections
                sections = [s[:-4] for s in sections]
            else:
                raise ValueError(f"Wall material ({len(os.listdir(wallmat_log_dir))}) and condition log directories have different number of sections ({len(os.listdir(condition_log_dir))}).")
            
            for section in tqdm(sections):
                wallmat_pth = torch.load(f"{wallmat_log_dir}/{section}.pth", weights_only=False)
                if len(wallmat_pth) == 0:
                    continue

                if '.png' in list(wallmat_pth.keys())[0]:
                    print("Detected old format wall material pth. Converting to new format...")
                    new_wallmat_pth = {}
                    for k, v in wallmat_pth.items():
                        new_k = k.split('/')[-1][:-4].split('_')[-1]
                        new_wallmat_pth[new_k] = v
                    wallmat_pth = new_wallmat_pth

                condition_pth = torch.load(f"{condition_log_dir}/{section}.pth", weights_only=False)
                # print(f"Processing {state} {group} {county} {section}...")
                if os.path.exists(f"{metadata_log_dir}/{section}.pth"):
                    print(f"Loading from metadata log directory {metadata_log_dir}...")
                    metadata_pth = torch.load(f"{metadata_log_dir}/{section}.pth", weights_only=False)
                else:
                    print(f"Loading from packed cropped directory {gsv_proc_packed_cropped_dir}...")
                    metadata_pth = {}
                    packed_cropped_path = f"{gsv_proc_packed_cropped_dir}/{section}.zip"
                    with zipfile.ZipFile(packed_cropped_path, 'r') as zip_ref:
                        metadata_pths = [n for n in zip_ref.namelist() if n.endswith('.pth')]
                        for pth_path in metadata_pths:
                            try:
                                with zip_ref.open(pth_path) as pth_file:
                                    pth = torch.load(pth_file, weights_only=False)
                                    if isinstance(pth, Detections):
                                        metadata = {
                                            'xyxy': pth.xyxy,
                                            'confidence': pth.confidence,
                                            'class_id': pth.class_id,
                                        }
                                    else:
                                        metadata = pth
                            except Exception as e:
                                print(f"Error loading cropped metadata {pth_path}: {e}")
                                exit()
                            bid = pth_path[:-4].split('_')[-1]
                            metadata_pth[bid] = metadata

                            

                    # os.makedirs(metadata_log_dir, exist_ok=True)
                    # torch.save(metadata_pth, f"{metadata_log_dir}/{section}.pth")
                    # print(f"Saved metadata pth to {metadata_log_dir}/{section}.pth")
                # print(metadata_pth.keys())
                # print(wallmat_pth.keys())
                # print(wallmat_pth)
                # print(condition_pth.keys())
                # exit()
                miss_count = 0
                print(f"Length of metadata_pth: {len(metadata_pth)}")
                print(f"Length of wallmat_pth: {len(wallmat_pth)}")
                print(f"Length of condition_pth: {len(condition_pth)}")

                for bid in tqdm(metadata_pth.keys()):
                    if bid not in state_nsi_bev:
                        print(f"Bid {bid} not found in NSI data for {state} {group} {county} {section}. Skipping.")
                        continue
                    # state_nsi_bev[bid]['cropping_metadata'] = metadata_pth[bid]

                    try:
                        if 'wall_material' in wallmat_pth[bid]:
                            state_nsi_bev[bid]['wall_material'] = wallmat_pth[bid]['wall_material'] # Indiana
                        elif 'material' in wallmat_pth[bid]:
                            state_nsi_bev[bid]['wall_material'] = wallmat_pth[bid]['material']
                        else:
                            raise KeyError(f"Wall material key not found for bid {bid} in {state} {group} {county} {section}.")
                        
                    
                        state_nsi_bev[bid]['condition'] = condition_pth[bid]['score']

                        state_metadata[bid] = {
                            'state': state,
                            'group': group,
                            'county': county,
                            'section': section,
                            'bid': bid,
                            'cropping': metadata_pth[bid],
                            'wall_material': wallmat_pth[bid],
                            'condition': condition_pth[bid],
                        }
                    except KeyError as e:
                        print(f"Missing data for bid {bid} in {state} {group} {county} {section}: {e}")
                        miss_count += 1
                        continue
                
                    # print(state_nsi_bev[bid])
                    # exit()

                print(f"Finished processing {state} {group} {county} {section}. Missing data for {miss_count}/{len(metadata_pth)} buildings.")

                

                # state_metadata.update(metadata_pth)
                # state_wallmat.update(wallmat_pth)
                # state_condition.update(condition_pth)

    torch.save(state_metadata, f"{gsv_bev_state_dir}/metadata.pth")

    with open(f"{gsv_bev_state_dir}/nsi_bev.json", 'w') as f:
        json.dump(state_nsi_bev, f, indent=4)


    # torch.save(state_metadata, f"{gsv_bev_state_dir}/building_metadata.pth")
    # torch.save(state_wallmat, f"{gsv_bev_state_dir}/wallmat.pth")
    # torch.save(state_condition, f"{gsv_bev_state_dir}/condition.pth")


def create_state_nsi(state, gsv_dl_dir, gsv_proc_dir, gsv_bev_dir):
    gsv_dl_state_dir = f"{gsv_dl_dir}/{state}"
    gsv_proc_state_dir = f"{gsv_proc_dir}/{state}"
    gsv_bev_state_dir = f"{gsv_bev_dir}/{state}"
    groups = os.listdir(gsv_dl_state_dir)

    groups = sorted(groups, key=lambda x: int(x.split('_')[-2]))
    print(groups)
    os.makedirs(gsv_bev_dir, exist_ok=True)
    os.makedirs(f"{gsv_bev_dir}/{state}", exist_ok=True)

    state_nsi = {}

    # nsi_file_path = f"{gsv_bev_state_dir}/nsi.json"

    # if os.path.exists(nsi_file_path):
    #     with open(nsi_file_path, 'r') as f:
    #         state_nsi = json.load(f)
    #     print(f"Loaded existing NSI data for {len(state_nsi)} buildings.")

    
    for group in groups:
        # if group.split('_')[-2] < 180:
        #     continue
        counties = os.listdir(f"{gsv_proc_state_dir}/{group}")
        print(counties)
        for county in counties:
            print(f"Processing NSI for {state} {group} {county}...")
            
            nsi_data_csv = f"{gsv_proc_state_dir}/{group}/{county}/nsi_data.csv"

            if not os.path.exists(nsi_data_csv):
                nsi_data_csv = f"{gsv_dl_state_dir}/{group}/{county}/nsi_data.csv"

            df = pd.read_csv(nsi_data_csv)
            cols = df.columns.tolist()

            for index, row in df.iterrows():
                entry = {}
                for col in cols:
                    entry[col] = row[col] if not pd.isna(row[col]) else None

                entry['state'] = state
                entry['group'] = group
                entry['county'] = county

                bid = entry['bid']

                state_nsi[bid] = entry

            

            print(f"Loaded NSI data for {len(state_nsi)} buildings so far.")
            
            # with open(f"{gsv_bev_state_dir}/nsi.json", 'r') as f:
            #     data = json.load(f)
            #     print(f"Loaded NSI data for {len(data)} buildings.")

    with open(f"{gsv_bev_state_dir}/nsi.json", 'w') as f:
        json.dump(state_nsi, f, indent=4)
        

            

    
