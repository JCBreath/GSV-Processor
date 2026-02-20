import os
import sys
import torch
import zipfile
from PIL import Image
from io import BytesIO
from modules.building_processor import BuildingProcessor
from tqdm import tqdm
import argparse
from supervision import Detections

if __name__ == "__main__":
        

    gsv_dl_dir = "GSV-Download"
    gsv_proc_dir = "GSV-Process"
    gsv_bev_dir = "GSV-Results"

    args = argparse.ArgumentParser()
    args.add_argument("--state", type=str, help="State to process")
    args.add_argument("--group_start", type=int, default=None, help="Starting group number")
    args.add_argument("--group_end", type=int, default=None, help="Ending group number")
    args.add_argument("--selected_county", type=str, default=None, help="Specific county to process")
    args.add_argument("--section_start", type=int, default=None, help="Starting section number")
    args.add_argument("--section_end", type=int, default=None, help="Ending section number")

    args.add_argument("--building_detect", action="store_true", help="Whether to perform building detection and cropping")
    args.add_argument("--wallmat", action="store_true", help="Whether to perform wall material classification")
    args.add_argument("--condition", action="store_true", help="Whether to perform building condition assessment")
    args.add_argument("--verbose", action="store_true", help="Whether to enable verbose output")
    args.add_argument("--skip", action="store_true", help="Whether to skip already processed files")

    args.add_argument("--bev", action="store_true", help="Export BEV CSV files of this state.")
    args.add_argument("--nsi", action="store_true", help="Export NSI JSON file of this state.")

    args = args.parse_args()

    state = args.state
    group_start = args.group_start
    group_end = args.group_end
    selected_county = args.selected_county
    section_start = args.section_start
    section_end = args.section_end
   
    building_detect = args.building_detect
    wallmat = args.wallmat
    condition = args.condition
    verbose = args.verbose
    skip = args.skip
    bev = args.bev
    nsi = args.nsi

    if not bev and not nsi:
        processor = BuildingProcessor()

    gsv_dl_state_dir = f"{gsv_dl_dir}/{state}"
    gsv_proc_state_dir = f"{gsv_proc_dir}/{state}"
    os.makedirs(gsv_proc_state_dir, exist_ok=True)

    if nsi:
        from modules.bev import create_state_nsi
        create_state_nsi(state, gsv_dl_dir, gsv_proc_dir, gsv_bev_dir)
        exit()

    if bev:
        from modules.bev import create_state_bevs
        create_state_bevs(state, gsv_proc_dir, gsv_bev_dir)
        exit()

    if group_start is None:
        print("Processing all groups.")
        if building_detect:
            groups = os.listdir(gsv_dl_state_dir)
        else:
            groups = os.listdir(gsv_proc_state_dir)
    elif group_end is None:
        print(f"Processing from group {group_start} onwards.")
        if building_detect:
            groups = os.listdir(gsv_dl_state_dir)
        else:
            groups = os.listdir(gsv_proc_state_dir)
        groups = [g for g in groups if int(g.split('_')[1]) >= group_start]
    else:
        print(f"Processing groups from {group_start} to {group_end}.")
        groups = [f"county_{i}_{i+9}" for i in range(group_start, group_end + 1, 10)]
    
    groups = sorted(groups, key=lambda x: int(x.split('_')[1]))
    print(f"{len(groups)} groups to process:")
    print(groups)

    for group in groups:
        gsv_dl_group_dir = f"{gsv_dl_state_dir}/{group}"
        gsv_proc_group_dir = f"{gsv_proc_state_dir}/{group}"
        os.makedirs(gsv_proc_group_dir, exist_ok=True)

        print(f"Processing group: {group}")

        if building_detect:
            countys = os.listdir(gsv_dl_group_dir)
        else:
            countys = os.listdir(gsv_proc_group_dir)
        for county in countys:
            if selected_county is not None and county != selected_county:
                continue

            gsv_dl_county_dir = f"{gsv_dl_group_dir}/{county}"
            gsv_proc_county_dir = f"{gsv_proc_group_dir}/{county}"
            os.makedirs(gsv_proc_county_dir, exist_ok=True)

            print(f"Processing county: {county}")

            gsv_dl_packed_building_views_dir = f"{gsv_dl_county_dir}/packed_building_views"
            gsv_proc_packed_cropped_dir = f"{gsv_proc_county_dir}/packed_cropped"
            os.makedirs(gsv_proc_packed_cropped_dir, exist_ok=True)

            metadata_log_dir = f"{gsv_proc_county_dir}/log_cropped_building"
            os.makedirs(metadata_log_dir, exist_ok=True)
            wallmat_log_dir = f"{gsv_proc_county_dir}/log_wall_material"
            os.makedirs(wallmat_log_dir, exist_ok=True)
            condition_log_dir = f"{gsv_proc_county_dir}/log_building_condition"
            os.makedirs(condition_log_dir, exist_ok=True)


            if building_detect:
                packed_building_view_zip_files = os.listdir(gsv_dl_packed_building_views_dir)
            else:
                packed_building_view_zip_files = os.listdir(gsv_proc_packed_cropped_dir)
            for packed_building_view_zip_file in packed_building_view_zip_files:
                section = packed_building_view_zip_file.split('.')[0]

                if section_start is not None:
                    if int(section.split('-')[-1]) < section_start:
                        print(f"Skipping section {section} before start {section_start}.")
                        continue
                
                packed_building_view_zip_path = f"{gsv_dl_packed_building_views_dir}/{packed_building_view_zip_file}"
                packed_cropped_path = f"{gsv_proc_packed_cropped_dir}/{packed_building_view_zip_file}"

                print(f"Processing: {packed_building_view_zip_path}")

                section_processed_results = []
                section_cropped_results = []

                if skip:
                    # print(packed_cropped_path, wallmat_log_dir, condition_log_dir, section)
                    if os.path.exists(packed_cropped_path) and os.path.exists(f"{wallmat_log_dir}/{section}.pth") and os.path.exists(f"{condition_log_dir}/{section}.pth"):
                        print(f"Skipping section {section} as all logs exist.")
                        continue

                # Detect and crop buildings
                if building_detect:
                    

                    with zipfile.ZipFile(packed_building_view_zip_path, 'r') as zip_ref:
                        name_list = zip_ref.namelist()
                        for file_name in tqdm(name_list) if verbose else name_list:
                            if not file_name.endswith('.jpg') and not file_name.endswith('.png'):
                                continue
                            # print(file_name)
                            try:
                                with zip_ref.open(file_name) as image_file:
                                    image = Image.open(BytesIO(image_file.read())).convert("RGB")
                            except Exception as e:
                                print(f"Error loading image {file_name}: {e}")
                                continue
                            
                            cropped_result = processor.detect_and_crop(image)
                            
                            
                            if cropped_result is not None:
                                section_cropped_results.append({
                                    'file_name': file_name.split('/')[-1][:29],
                                    'bid': file_name[:-4].split('_')[-1],
                                    'cropped_image': cropped_result['result'],
                                    'metadata': cropped_result['metadata']
                                })

                    print(f"Section building cropping complete. Found {len(section_cropped_results)} buildings.")

                    with zipfile.ZipFile(packed_cropped_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                        for item in tqdm(section_cropped_results) if verbose else section_cropped_results:
                            file_name = item['file_name']
                            bid = item['bid']
                            cropped_image = item['cropped_image']
                            cropped_image_bytes = BytesIO()
                            cropped_image.save(cropped_image_bytes, format='JPEG')
                            cropped_image_bytes.seek(0)
                            zip_out.writestr(f"{file_name}_{bid}.png", cropped_image_bytes.read())
                    print(f"Cropped images saved to: {packed_cropped_path}")

                # Load cropped images for further processing
                else:
                    with zipfile.ZipFile(packed_cropped_path, 'r') as zip_ref:
                        name_list = zip_ref.namelist()
                        cropped_images = [n for n in name_list if n.endswith('.jpg') or n.endswith('.png')]

                        if os.path.exists(f"{metadata_log_dir}/{section}.pth"):
                            section_metadata_pth = torch.load(f"{metadata_log_dir}/{section}.pth", weights_only=False)
                        else:
                            section_metadata_pth = None

                        for cropped_image in tqdm(cropped_images) if verbose else cropped_images:
                            try:
                                with zip_ref.open(cropped_image) as image_file:
                                    image = Image.open(BytesIO(image_file.read())).convert("RGB")

                                if section_metadata_pth is not None:
                                    bid = cropped_image[:-4].split('_')[-1]
                                    metadata = section_metadata_pth[bid]
                                    
                                else:
                                    with zip_ref.open(cropped_image.replace('gsv_cropped_building','log_cropped_building')[:-4] + '.pth') as pth_file:
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
                                print(f"Error loading cropped image {cropped_image}: {e}")
                                exit()
                            
                            section_cropped_results.append({
                                'file_name': cropped_image.split('/')[-1][:29],
                                'bid': cropped_image[:-4].split('_')[-1],
                                'cropped_image': image,
                                'metadata': metadata
                            })
                            # print(pth.xyxy)
                            # print(section_cropped_results)
                            # exit()
                        # print(len([n for n in name_list if n.endswith('.jpg') or n.endswith('.pth')]))
                # exit()
                        

                section_metadata_results = {}
                section_wallmat_results = {}
                section_condition_results = {}


                for item in tqdm(section_cropped_results) if verbose else section_cropped_results:
                    file_name = item['file_name']
                    bid = item['bid']
                    cropped_image = item['cropped_image']
                    metadata = item['metadata']

                    

                    section_metadata_results[bid] = {
                        'file_name': file_name,
                        'xyxy': metadata['xyxy'],
                        'confidence': metadata['confidence'],
                        'class_id': metadata['class_id'],
                    }

                    if wallmat:
                        wallmat_result = processor.get_wallmat(cropped_image)
                        section_wallmat_results[bid] = wallmat_result

                    if condition:
                        condition_result = processor.get_condition(cropped_image)
                        section_condition_results[bid] = condition_result

                if building_detect or not os.path.exists(f"{metadata_log_dir}/{section}.pth"):
                    torch.save(section_metadata_results, f"{metadata_log_dir}/{section}.pth")
                if wallmat:
                    torch.save(section_wallmat_results, f"{wallmat_log_dir}/{section}.pth")
                if condition:
                    torch.save(section_condition_results, f"{condition_log_dir}/{section}.pth")

                print("Processing complete.")

                


