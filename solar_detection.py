import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import shutil
import random
import yaml
from ultralytics import YOLO

## User defined functions

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # Delete the folder and all its contents
    os.makedirs(path)         # Recreate the empty folder


## CONFIG folders

im_folder = "F:/Image Datasets/Averros/Solar Segmentation"
CLASS_NAMES = ['Panel']  # Replace with your actual class names
COCO_JSON_PATH = f'{im_folder}/panel_seg_coco.json'  # COCO JSON file
IMAGE_DIR = f'{im_folder}/images/'  # Folder with original images
YOLO_LABEL_DIR = f'{im_folder}/labels_yolo/' # dir containing yolo labels
YOLO_LABELED_IM_DIR = f'{im_folder}/annotated_yolo_images/' # Folder to save annotated Yolo images to cross verify labels
DATASET_DIR = f'{im_folder}/dataset/' #create image dataset for Yolo training

TRAIN_RATIO = 0.6 # Train = 60%, Test = 40%, due to small dataset require more images for validation,

os.makedirs(YOLO_LABELED_IM_DIR, exist_ok=True)
os.makedirs(YOLO_LABEL_DIR, exist_ok=True)


def prepare_data():
    for split in ['train', 'val']:
        reset_folder(os.path.join(DATASET_DIR, split, 'images'))
        reset_folder(os.path.join(DATASET_DIR, split, 'labels'))

    ## Load COCO JSON and convert them into labels required for Yolo

    print("\n Converting JSON into .txt labels for Yolo........")
    with open(COCO_JSON_PATH, 'r') as f:
        coco = json.load(f)

    # Build lookup tables
    image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann)

    # Convert and save
    for img_id, anns in tqdm(annotations_by_image.items()):
        img_name = os.path.splitext(image_id_to_filename[img_id])[0]
        img_w, img_h = image_id_to_size[img_id]
        label_path = os.path.join(YOLO_LABEL_DIR, f"{img_name}.txt")

        lines = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            class_id = category_id_to_index[ann['category_id']]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))


    ##


    # Created annotated images to verify labels
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Process each image
    print("\n Creating Annotated images from Labels for verification............")
    for image_file in tqdm(os.listdir(IMAGE_DIR)):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(YOLO_LABEL_DIR, f"{base_name}.txt")
        image_path = os.path.join(IMAGE_DIR, image_file)

        if not os.path.exists(label_path):
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                x_center *= img_w
                y_center *= img_h
                w *= img_w
                h *= img_h
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2

                draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
                label = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else str(class_id)
                draw.text((x1+20, y1 + 10), label, fill="cyan", font=font)

        output_path = os.path.join(YOLO_LABELED_IM_DIR, f"Y_{image_file}")
        image.save(output_path)

    ##

    print("\n Creating train / Val sets and data.yaml ............")
    # Get all image files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    # Split into train and val
    split_index = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def copy_files(file_list, split):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"

            # Copy image
            src_img = os.path.join(IMAGE_DIR, img_file)
            dst_img = os.path.join(DATASET_DIR, split, 'images', img_file)
            shutil.copy2(src_img, dst_img)

            # Copy label
            src_lbl = os.path.join(YOLO_LABEL_DIR, label_file)
            dst_lbl = os.path.join(DATASET_DIR, split, 'labels', label_file)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    # Create data.yaml
    data_yaml = {
        'train': os.path.join(DATASET_DIR, 'train/images'),
        'val': os.path.join(DATASET_DIR, 'val/images'),
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("✅ Dataset prepared and data.yaml created.")

def train_yolo():
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 4
    MODEL_NAME = "yolo11n.pt"
    model = YOLO(MODEL_NAME)
    model.train(
        data=os.path.join(DATASET_DIR, 'data.yaml'),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE
    )


if __name__ == "__main__":
    print("\n 🚀 Preparing data for YOLO...")
    prepare_data()
    print("\n 🚀 Starting YOLOv11 training...")
    train_yolo()
