import zipfile
import os
import urllib.request


def download_and_unpack_raw_datasets(
        download_url_list: list,
        dir_to_unpack: str
):
    # Removing old file if necessary
    for file in os.listdir(dir_to_unpack):
        os.remove(os.path.join(dir_to_unpack, file))

    # Downloading and unzipping files
    for url in download_url_list:
        filename = url.split(sep='/')[-1]
        full_file_path = dir_to_unpack + filename
        urllib.request.urlretrieve(url, full_file_path)

        with zipfile.ZipFile(full_file_path) as zip_file:
            zip_file.extractall(dir_to_unpack)

        os.remove(full_file_path)

    # Removing unnecessary subfolders
    for name in os.listdir(dir_to_unpack):
        subfolder_path = os.path.join(dir_to_unpack, name)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                os.replace(
                    os.path.join(subfolder_path, file),
                    os.path.join(dir_to_unpack, file)
                )

            os.rmdir(subfolder_path)

