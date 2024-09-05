import zipfile
import os
import urllib.request


def download_and_unpack_raw_datasets(
        download_url_list: list,
        dir_to_unpack: str
):
    for url in download_url_list:
        filename = url.split(sep='/')[-1]
        full_file_path = dir_to_unpack + filename
        urllib.request.urlretrieve(url, full_file_path)

        with zipfile.ZipFile(full_file_path) as zip_file:
            zip_file.extractall(dir_to_unpack)

        os.remove(full_file_path)

