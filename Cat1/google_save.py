from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def authenticate():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive


def create_directory(drive, parent_id, folder_name):
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': parent_id}]
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder


def upload_files_in_directory(drive, parent_id, local_directory):
    for item in os.listdir(local_directory):
        item_path = os.path.join(local_directory, item)

        if os.path.isfile(item_path):
            file = drive.CreateFile({
                'title': item,
                'parents': [{'id': parent_id}]
            })
            file.Upload()
            # Uploading files to google drive
            print("Uploading - " + f"{item}")
        elif os.path.isdir(item_path):
            folder = create_directory(drive, parent_id, item)
            upload_files_in_directory(drive, folder['id'], item_path)


def upload_to_gdrive(source_directory, destination_folder_name, parent_folder_name):
    source_directory = source_directory  # Replace with your source folder path
    destination_folder_name = destination_folder_name  # Replace with your desired destination folder name
    parent_folder_name = parent_folder_name  # Replace with your desired parent folder name
    drive = authenticate()
    parent_folder = create_directory(drive, "root", parent_folder_name)
    destination_folder = create_directory(drive, parent_folder['id'], destination_folder_name)
    upload_files_in_directory(drive, destination_folder['id'], source_directory)

    print("Upload completed!")