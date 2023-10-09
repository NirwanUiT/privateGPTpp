import os
import time
import shutil

dir_path = "/data/privateGPTpp/source_documents/"
db_path = "/data/privateGPTpp/db"

if os.listdir(dir_path):
    print("Directory is not empty. Deleting contents in 5 minutes...")
    time.sleep(60) # wait for 5 minutes
    # Delete the db folder in db_path
    try:
        shutil.rmtree(db_path)
        print(f"Deleted {db_path}")
    except Exception as e:
        print(f"Error deleting {db_path}: {e}")
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            


else:
    print("Directory is empty.")
