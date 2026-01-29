import os
from supervision import Detections
import zipfile
import torch
from tqdm import tqdm

def create_state_bevs(state, gsv_proc_dir, gsv_bev_dir):
    gsv_proc_state_dir = f"{gsv_proc_dir}/{state}"
    gsv_bev_state_dir = f"{gsv_bev_dir}/{state}"
    groups = os.listdir(gsv_proc_state_dir)

    print(groups)
    os.makedirs(gsv_bev_dir, exist_ok=True)
    os.makedirs(f"{gsv_bev_dir}/{state}", exist_ok=True)

    state_metadata = {}
    state_wallmat = {}
    state_condition = {}

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

            if len(os.listdir(wallmat_log_dir)) == len(os.listdir(condition_log_dir)):
                sections = os.listdir(wallmat_log_dir)
                sections = [s[:-4] for s in sections]
            else:
                raise ValueError("Wall material and condition log directories have different number of sections.")
            
            for section in tqdm(sections):
                # print(f"Processing {state} {group} {county} {section}...")
                if os.path.exists(f"{metadata_log_dir}/{section}.pth"):
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
                    os.makedirs(metadata_log_dir, exist_ok=True)
                    torch.save(metadata_pth, f"{metadata_log_dir}/{section}.pth")
                    print(f"Saved metadata pth to {metadata_log_dir}/{section}.pth")
                wallmat_pth = torch.load(f"{wallmat_log_dir}/{section}.pth", weights_only=False)
                condition_pth = torch.load(f"{condition_log_dir}/{section}.pth", weights_only=False)

                state_metadata.update(metadata_pth)
                state_wallmat.update(wallmat_pth)
                state_condition.update(condition_pth)
        
    torch.save(state_metadata, f"{gsv_bev_state_dir}/all_metadata.pth")
    torch.save(state_wallmat, f"{gsv_bev_state_dir}/all_wallmat.pth")
    torch.save(state_condition, f"{gsv_bev_state_dir}/all_condition.pth")




            