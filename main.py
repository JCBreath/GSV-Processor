import os
import sys
import torch
import zipfile
from PIL import Image
from io import BytesIO
from modules.building_processor import BuildingProcessor
from tqdm import tqdm

if __name__ == "__main__":
    processor = BuildingProcessor()

    gsv_dl_dir = "GSV-Download"
    gsv_proc_dir = "GSV-Process"

    state = sys.argv[1]
    group_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    group_end = int(sys.argv[3]) if len(sys.argv) > 3 else None
    selected_county = sys.argv[4] if len(sys.argv) > 4 else None
    verbose = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    gsv_dl_state_dir = f"{gsv_dl_dir}/{state}"
    gsv_proc_state_dir = f"{gsv_proc_dir}/{state}"
    os.makedirs(gsv_proc_state_dir, exist_ok=True)

    

    if group_start is None:
        print("Processing all groups.")
        groups = os.listdir(gsv_dl_state_dir)
    elif group_end is None:
        print(f"Processing from group {group_start} onwards.")
        groups = os.listdir(gsv_dl_state_dir)
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

        countys = os.listdir(gsv_dl_group_dir)
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

            packed_building_view_zip_files = os.listdir(gsv_dl_packed_building_views_dir)
            for packed_building_view_zip_file in packed_building_view_zip_files:
                section = packed_building_view_zip_file.split('.')[0]
                packed_building_view_zip_path = f"{gsv_dl_packed_building_views_dir}/{packed_building_view_zip_file}"
                packed_cropped_path = f"{gsv_proc_packed_cropped_dir}/{packed_building_view_zip_file}"

                print(f"Processing: {packed_building_view_zip_path}")

                section_processed_results = []
                section_cropped_results = []

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

                section_metadata_results = {}
                section_wallmat_results = {}
                section_condition_results = {}


                for item in tqdm(section_cropped_results) if verbose else section_cropped_results:
                    file_name = item['file_name']
                    bid = item['bid']
                    cropped_image = item['cropped_image']
                    metadata = item['metadata']

                    wallmat_result = processor.get_wallmat(cropped_image)
                    condition_result = processor.get_condition(cropped_image)

                    section_metadata_results[bid] = {
                        'file_name': file_name,
                        'xyxy': metadata['xyxy'],
                        'confidence': metadata['confidence'],
                        'class_id': metadata['class_id'],
                    }

                    section_wallmat_results[bid] = wallmat_result
                    section_condition_results[bid] = condition_result

                
                torch.save(section_metadata_results, f"{metadata_log_dir}/{section}.pth")
                torch.save(section_wallmat_results, f"{wallmat_log_dir}/{section}.pth")
                torch.save(section_condition_results, f"{condition_log_dir}/{section}.pth")

                print("Processing complete.")

                


