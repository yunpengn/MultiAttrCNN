import errno
import os
import random
import shutil

data_dir = "LFW_data"
extract_dir = "LFW_extract"
label_dir = "LFW_label"
label_male_names = "male_names.txt"
label_female_names = "female_names.txt"


def convert_name_to_path(name):
    # Split by last occurrence of "_"
    split = name.rsplit("_", 1)
    return os.path.join(data_dir, split[0], name)


def copy_file(src, dest):
    shutil.copyfile(src, dest)


def create_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            # To deal with potential racing conditions.
            if e.errno != errno.EEXIST:
                raise


def is_train(rate=0.2):
    ran = random.uniform(0, 1)
    return ran < 1 - rate


def extract_gender(label_file, extract_train_folder, extract_val_folder, amount=-1):
    # Removes the old extracted data
    shutil.rmtree(extract_train_folder, ignore_errors=True)
    shutil.rmtree(extract_val_folder, ignore_errors=True)

    # Create the relevant directories.
    create_folder(extract_train_folder)
    create_folder(extract_val_folder)

    with open(label_file) as f:
        content = [line.strip().rstrip('\n') for line in f.readlines()]
        print("Have read %s lines from file at %s." % (len(content), label_file))

        for i, line in enumerate(content):
            # Stops the extract if has already reached the limit.
            if amount != -1 and i > amount:
                break

            image_path = convert_name_to_path(line)
            if not os.path.isfile(image_path):
                print("WARNING: the image at %s does not exist." % image_path)
                continue

            if is_train():
                dest_path = os.path.join(extract_train_folder, line)
            else:
                dest_path = os.path.join(extract_val_folder, line)
            copy_file(image_path, dest_path)

            if i % 100 == 0:
                print("Have handled %d images." % i)

        print("Finished the processing on the file at %s." % label_file)


# Tries to extract the same amount of data for both genders.
extract_gender(os.path.join(label_dir, label_male_names),
               os.path.join(extract_dir, "train/male"), os.path.join(extract_dir, "val/male"), 3300)
extract_gender(os.path.join(label_dir, label_female_names),
               os.path.join(extract_dir, "train/female"), os.path.join(extract_dir, "val/female"))
