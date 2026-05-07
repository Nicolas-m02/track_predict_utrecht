#%%

import os
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import datetime
import pydicom
from pydicom.valuerep import DSfloat, IS
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    generate_uid,
)
import MR_TC_pb2 as mrtc


STUDY_UID = generate_uid()
FRAME_OF_REF_UID = generate_uid()
SERIES_UID = generate_uid()

# 
# ---------- image conversion ----------
def mrtc_to_numpy(img):
    slope = img.rescale_function.slope
    intercept = img.rescale_function.intercept
    bpp = img.bytes_per_pixel
    bitspp = img.bits_per_pixel
    rows = img.rows
    cols = img.columns
    buf = img.pixel_data

    out = np.empty((rows, cols), dtype=np.float32)

    fmt_map = {
        1: "B",
        2: "H",
        4: "I",
    }
    if bpp not in fmt_map:
        raise ValueError(f"Unsupported bytes_per_pixel: {bpp}")

    fmt = "<" + fmt_map[bpp]

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            offset = index * bpp
            value = struct.unpack_from(fmt, buf, offset)[0]
            if bitspp < bpp * 8:
                value &= (1 << bitspp) - 1
            out[row, col] = intercept + slope * float(value)

    return out

# ---- save numpy as dicom with geometry ----
def save_numpy_as_dicom_with_geometry(
    image,
    out_path,
    slope,
    intercept,
    instance_number,
    pixel_spacing=(2.604, 2.604),
    slice_thickness=20.0,
    row_dir=(0.0, 1.0, 0.0),
    col_dir=(0.0, 0.0, -1.0),
    upper_left_voxel=(0.0, -83.69, 123.69),
    slice_index=0,
):
    # ---------- File Meta ----------
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "PYDICOM_MR_1"

    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # ---------- Dates ----------
    now = datetime.datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.StudyDate = ds.ContentDate
    ds.StudyTime = ds.ContentTime

    # ---------- Identifiers ----------
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.FrameOfReferenceUID = FRAME_OF_REF_UID
    ds.InstanceNumber = IS(instance_number)

    # ---------- Patient / Study ----------
    ds.Modality = "MR"
    ds.PatientName = "Anonymous"
    ds.PatientID = "000000"
    ds.PatientSex = "O"
    ds.StudyID = "1"
    ds.StudyDescription = "MR Generated"
    ds.SeriesDescription = "MR Slice Series"

    # ---------- Image ----------
    rows, cols = image.shape
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    ds.RescaleSlope = DSfloat(slope)
    ds.RescaleIntercept = DSfloat(intercept)

    # ---------- Geometry ----------
    ds.PixelSpacing = [DSfloat(pixel_spacing[0]), DSfloat(pixel_spacing[1])]
    ds.SliceThickness = DSfloat(slice_thickness)
    ds.SpacingBetweenSlices = DSfloat(slice_thickness)

    ds.ImageOrientationPatient = [
        DSfloat(row_dir[0]), DSfloat(row_dir[1]), DSfloat(row_dir[2]),
        DSfloat(col_dir[0]), DSfloat(col_dir[1]), DSfloat(col_dir[2]),
    ]

    ipp_z = upper_left_voxel[2] - slice_index * slice_thickness
    ds.ImagePositionPatient = [
        DSfloat(upper_left_voxel[0]),
        DSfloat(upper_left_voxel[1]),
        DSfloat(ipp_z),
    ]

    ds.SliceLocation = DSfloat(ipp_z)

    # ---------- MR specific ----------
    ds.MRAcquisitionType = "2D"
    ds.ScanningSequence = "GR"
    ds.SequenceVariant = "NONE"
    ds.ScanOptions = "NONE"
    ds.MagneticFieldStrength = DSfloat(1.5)
    ds.ImagingFrequency = DSfloat(63.9)
    ds.RepetitionTime = DSfloat(1000.0)
    ds.EchoTime = DSfloat(10.0)
    ds.FlipAngle = DSfloat(90.0)

    # ---------- Pixel Data ----------
    stored = (image - intercept) / slope
    stored = np.clip(stored, 0, 65535).astype(np.uint16)
    ds.PixelData = stored.tobytes()

    ds.save_as(out_path, write_like_original=False)


# ---------- constants ----------
HEADER_SIZE = 4 * 4
TRAILER_SIZE = 4

folder = "/utrecht_exp/data/all_dat_files/dat_data/"
out_dir = "/utrecht_exp/data/dicom_output_20260505"
os.makedirs(out_dir, exist_ok=True)

file_list = sorted([f for f in os.listdir(folder) if f.endswith(".dat")])

vmin = 0
vmax = 12000

series_uid = generate_uid()
instance = 1


# ---------- display ----------
plt.ion()
fig, ax = plt.subplots()
im = None


# ---------- main loop ----------
for filename in file_list:
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as f:
        buffer = f.read()

    size, major, minor, type_ = struct.unpack_from("<4I", buffer, offset=0)
    payload_size = size - HEADER_SIZE - TRAILER_SIZE
    payload = buffer[HEADER_SIZE : HEADER_SIZE + payload_size]

    if type_ != mrtc.MessageType.MESSAGE_TYPE_IMAGE_DATA:
        continue

    img_msg = mrtc.ImageDataMessage()
    try:
        img_msg.ParseFromString(payload)
    except Exception as e:
        print(f"failed to parse {filename}: {e}")
        continue

    image_array = mrtc_to_numpy(img_msg)

    # ---- display ----
    if im is None:
        im = ax.imshow(image_array, cmap="gray", vmin=vmin, vmax=vmax)
        #plt.colorbar(im, ax=ax)
    else:
        im.set_data(image_array)
        ax.relim()
        ax.autoscale_view()

    plt.draw()
    #plt.pause(0.01)

    # ---- save dicom ----
    dcm_path = os.path.join(out_dir, f"img_{instance:04d}.dcm")
    save_numpy_as_dicom_with_geometry(
        image=image_array,
        out_path=dcm_path,
        slope=img_msg.rescale_function.slope,
        intercept=img_msg.rescale_function.intercept,
        instance_number=instance,
        pixel_spacing=(2.604, 2.604),
        slice_thickness=20.0,
        row_dir=(0.0, 1.0, 0.0),
        col_dir=(0.0, 0.0, -1.0),
        upper_left_voxel=(0.0, -83.69, 123.69),
        slice_index=instance-1
    )

    instance += 1

plt.axis("off")
plt.ioff()
plt.show()
