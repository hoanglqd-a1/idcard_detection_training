import argparse
import json
import numpy as np
import os
import shutil
import zipfile
import zlib
import time
from tempfile import TemporaryDirectory
from PIL import Image
from glob import glob
from ftplib import FTP, all_errors, error_perm
from tqdm import tqdm
from pathlib import Path

download_links = [
                    'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip',
                    'ftp://smartengines.com/midv-500/dataset/03_aut_id_old.zip',
                    'ftp://smartengines.com/midv-500/dataset/04_aut_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/07_chl_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/08_chn_homereturn.zip',
                    'ftp://smartengines.com/midv-500/dataset/09_chn_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/10_cze_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/11_cze_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
                    'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
                    'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
                    'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
                    'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
                    'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip',
                    'ftp://smartengines.com/midv-500/dataset/18_dza_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/19_esp_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip',
                    'ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip',
                    'ftp://smartengines.com/midv-500/dataset/22_est_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/24_fin_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/26_hrv_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/28_hun_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/29_irn_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/30_ita_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/33_mac_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/34_mda_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/35_nor_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/36_pol_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/37_prt_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip',
                    'ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip',
                    'ftp://smartengines.com/midv-500/dataset/40_srb_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/42_svk_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/43_tur_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/44_ukr_id.zip',
                    'ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/46_ury_passport.zip',
                    'ftp://smartengines.com/midv-500/dataset/47_usa_bordercrossing.zip',
                    'ftp://smartengines.com/midv-500/dataset/48_usa_passportcard.zip',
                    'ftp://smartengines.com/midv-500/dataset/49_usa_ssn82.zip',
                    'ftp://smartengines.com/midv-500/dataset/50_xpo_id.zip'
                ]

DATADIR = Path(__file__).resolve().parent / 'datasets' / 'MIDV500'

SUBDIR = ['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'TA', 'TS', 'PA', 'PS']
HOST = 'smartengines.com'

def validate_zip(zip_path):
    """Check the archive directory and each member's CRC before caching it."""
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f'Corrupt archive member: {bad_member}')


def download_zipfile(file_name, save_dir, *, timeout=30, attempts=3):
    """Return a validated ZIP, retrying failed transfers from the beginning."""
    if Path(file_name).name != file_name or not file_name.endswith('.zip'):
        raise ValueError('file_name must be a ZIP filename without directories')
    if attempts < 1 or timeout <= 0:
        raise ValueError('attempts and timeout must be positive')
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    zip_path = save_dir / file_name
    partial_path = save_dir / (file_name + '.part')
    remote_path = '/midv-500/dataset/' + file_name

    if zip_path.is_file():
        try:
            validate_zip(zip_path)
            return zip_path
        except (zipfile.BadZipFile, EOFError, zlib.error):
            print(f'Replacing invalid cached archive: {zip_path}')

    for attempt in range(1, attempts + 1):
        try:
            with FTP(HOST, timeout=timeout) as ftp:
                ftp.login()
                ftp.voidcmd('TYPE I')
                try:
                    file_size = ftp.size(remote_path)
                except error_perm:
                    # Some servers reject SIZE while still allowing RETR.
                    file_size = None

                with partial_path.open('wb') as output, tqdm(
                    total=file_size, unit='B', unit_scale=True,
                    unit_divisor=1024, desc=f'Downloading {file_name}',
                ) as progress:
                    def write_chunk(data):
                        output.write(data)
                        progress.update(len(data))

                    ftp.retrbinary(
                        f'RETR {remote_path}', write_chunk,
                        blocksize=1024 * 1024,
                    )

            if file_size is not None and partial_path.stat().st_size != file_size:
                raise zipfile.BadZipFile(f'Incomplete download: {file_name}')
            validate_zip(partial_path)
            partial_path.replace(zip_path)
            return zip_path
        except (*all_errors, zipfile.BadZipFile, zlib.error) as error:
            if attempt == attempts:
                raise
            print(f'Download attempt {attempt}/{attempts} failed: {error}. Retrying...')
            time.sleep(min(2 ** (attempt - 1), 4))
        finally:
            # A failed transfer must never look like a completed ZIP.
            partial_path.unlink(missing_ok=True)


def extract_zipfile(zip_path, save_dir):
    """Publish an extracted dataset only after every member extracts successfully."""
    zip_path = Path(zip_path)
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    destination = save_dir / zip_path.stem
    if destination.resolve().parent != save_dir:
        raise ValueError(f'Extraction directory must be inside {save_dir}')

    with TemporaryDirectory(prefix=f'.{zip_path.stem}-', dir=save_dir) as temporary:
        staging = Path(temporary).resolve()
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            for member in members:
                if not (staging / member.filename).resolve().is_relative_to(staging):
                    raise ValueError(f'Archive member escapes extraction directory: {member.filename}')
            with tqdm(
                total=sum(member.file_size for member in members),
                unit='B', unit_scale=True, unit_divisor=1024,
                desc=f'Extracting {zip_path.name}',
            ) as progress:
                for member in members:
                    archive.extract(member, staging)
                    progress.update(member.file_size)

        extracted = staging / zip_path.stem
        if not all((extracted / name).is_dir() for name in ('images', 'ground_truth')):
            raise ValueError(f'Archive is missing the expected MIDV500 folders: {zip_path}')
        (extracted / '.extraction-complete').write_text('complete\n', encoding='utf-8')
        # Replace old, possibly incomplete extractions only after staging succeeds.
        if destination.exists():
            shutil.rmtree(destination)
        extracted.replace(destination)
    return destination


def prepare_archive(file_name, save_dir):
    """Reuse only extractions carrying a completion marker."""
    destination = Path(save_dir) / Path(file_name).stem
    if (destination / '.extraction-complete').is_file() and all(
        (destination / name).is_dir() for name in ('images', 'ground_truth')
    ):
        return destination
    zip_path = download_zipfile(file_name, save_dir)
    return extract_zipfile(zip_path, save_dir)


def read_image(img, annot):
    image = Image.open(img)
    orig_shape = image.size
    quad = json.load(open(annot, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    normalized_coords = coords.astype(np.float32)
    normalized_coords[:, 0] = normalized_coords[:, 0] / orig_shape[0]
    normalized_coords[:, 1] = normalized_coords[:, 1] / orig_shape[1]
    if not is_valid_annot(normalized_coords):
        return None, None
    annotation = "0 %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (normalized_coords[0, 0], normalized_coords[0, 1], normalized_coords[1, 0], normalized_coords[1, 1], normalized_coords[2, 0], normalized_coords[2, 1], normalized_coords[3, 0], normalized_coords[3, 1])
    return image, annotation

def is_valid_annot(annot: np.ndarray) -> bool:
    return ((annot < 1) & (annot > 0)).all()

def download_and_unzip(dataset_dir=DATADIR):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    IMAGE = os.path.join(dataset_dir, 'images')
    ANNOT = os.path.join(dataset_dir, 'labels')
    TEMP = os.path.join(dataset_dir, 'temp')
    for directory in (IMAGE, ANNOT):
        output_dir = Path(directory)
        if output_dir.resolve().parent != dataset_dir:
            raise ValueError(f'Output directory must be inside {dataset_dir}: {output_dir}')
        if output_dir.exists():
            shutil.rmtree(output_dir)
    for directory in (IMAGE, ANNOT, TEMP):
        os.makedirs(directory, exist_ok=True)

    file_idx = 0

    for link in download_links:
        filename = link.rsplit('/', 1)[-1]
        directory_name = str(prepare_archive(filename, TEMP))

        print('Prepare dataset... ', directory_name)
        img_dir_path = directory_name + '/images/'
        annot_dir_path = directory_name + '/ground_truth/'

        for subdir in SUBDIR:
            img_list = sorted(glob(img_dir_path + subdir + '/*.tif'))
            for i, img in enumerate(img_list):
                if i % 7 != 0: continue
                annot = Path(annot_dir_path) / subdir / (Path(img).stem + '.json')
                image, annotation = read_image(img, annot)
                if image is None: continue
                image.save(IMAGE + '/image' + str(file_idx) + '.png')
                with open(ANNOT + '/image' + str(file_idx) + '.txt', 'w') as f:
                    f.write(annotation)

                file_idx += 1

        print('----------------------------------------------------------------------')
    

def split_train_val(dataset_dir=DATADIR):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    IMAGE = os.path.join(dataset_dir, 'images')
    ANNOT = os.path.join(dataset_dir, 'labels')
    image_list = sorted(glob(IMAGE + '/*.png'))

    # create the train and val directories if they don't exist
    os.makedirs(os.path.join(IMAGE, 'train'), exist_ok=True)
    os.makedirs(os.path.join(IMAGE, 'val'), exist_ok=True)
    os.makedirs(os.path.join(ANNOT, 'train'), exist_ok=True)
    os.makedirs(os.path.join(ANNOT, 'val'), exist_ok=True)

    # shuffle and split the dataset into training and validation sets
    train_size = int(len(image_list) * 0.8)
    filename_list = [os.path.basename(img).split('.')[0] for img in image_list]
    np.random.default_rng(42).shuffle(filename_list)
    train_filenames = set(filename_list[:train_size])
    val_filenames = set(filename_list[train_size:])

    # move the training and validation images and annotations to their respective directories
    for filename in train_filenames:
        image_name = filename + '.png'
        annot_name = filename + '.txt'
        shutil.move(os.path.join(IMAGE, image_name), os.path.join(IMAGE, 'train', image_name))
        shutil.move(os.path.join(ANNOT, annot_name), os.path.join(ANNOT, 'train', annot_name))

    for filename in val_filenames:
        image_name = filename + '.png'
        annot_name = filename + '.txt'
        shutil.move(os.path.join(IMAGE, image_name), os.path.join(IMAGE, 'val', image_name))
        shutil.move(os.path.join(ANNOT, annot_name), os.path.join(ANNOT, 'val', annot_name))


def write_data_yaml(dataset_dir=DATADIR):
    """Write the Ultralytics YOLO configuration beside the images and labels."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / 'data.yaml'
    yaml_path.write_text(
        f'path: {json.dumps(dataset_dir.as_posix())}\n'
        'train: images/train\n'
        'val: images/val\n'
        '\n'
        'names:\n'
        '  0: card\n',
        encoding='utf-8',
    )
    return yaml_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare the MIDV500 dataset for YOLO OBB training.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DATADIR,
        help=f'Dataset directory (default: {DATADIR}). Existing images and labels are rebuilt.',
    )
    args = parser.parse_args(argv)
    dataset_dir = args.output_dir.expanduser().resolve()
    download_and_unzip(dataset_dir)
    split_train_val(dataset_dir)
    print(f'Dataset configuration saved to {write_data_yaml(dataset_dir)}')


if __name__ == '__main__':
    main()
