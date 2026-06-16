#%%
import SimpleITK as sitk
import os
sitk.ProcessObject_SetGlobalWarningDisplay(False)
from tqdm import tqdm


def convert_ima_to_mha(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for dir_name in tqdm(sorted(os.listdir(input_folder)), desc="Converting folders"):
        total_in_path = os.path.join(input_folder, dir_name)
        total_out_path = os.path.join(output_folder, dir_name)
        if not os.path.exists(total_out_path):
            os.makedirs(total_out_path)
        for filename in sorted(os.listdir(total_in_path)):
            if filename.endswith(".IMA"):
                try:
                    input_path = os.path.join(total_in_path, filename)
                    output_file = os.path.join(total_out_path,filename.replace(".IMA", ".mha"))
                    # Read the .ima file
                    image = sitk.ReadImage(input_path)
                    image = sitk.Cast(image, sitk.sitkInt16)
                    
                    # Write the image as .mha file
                    sitk.WriteImage(image, output_file)
                except:
                    print(f"Error converting {input_path} to {output_file}")
                    
        print(f"Converted {dir_name} from .IMA to .mha")

def single_conv(input_folder, output_folder):
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".dcm") and not filename.startswith("."):
            input_path = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder,filename.replace(".dcm", ".mha"))
            # Read the .ima file
            image = sitk.ReadImage(input_path)
            
            # Write the image as .mha file
            sitk.WriteImage(image, output_file)
    print(f"Converted {input_folder} from .dcm to .mha")


input_folder = "/utrecht_exp/data/dicom_output_20260505/"
output_folder = "/utrecht_exp/data/mha_converted/"
single_conv(input_folder, output_folder)
print("Conversion complete.")

