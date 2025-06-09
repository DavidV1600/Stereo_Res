import os
import xml.etree.ElementTree as ET

CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def convert_voc_to_yolo(xml_dir, output_dir, img_dir):
    os.makedirs(output_dir, exist_ok=True)
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        yolo_lines = []

        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in CLASS_NAMES:
                continue
            cls_id = CLASS_NAMES.index(cls_name)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format (cx, cy, w, h), normalized
            cx = (xmin + xmax) / 2 / image_width
            cy = (ymin + ymax) / 2 / image_height
            w = (xmax - xmin) / image_width
            h = (ymax - ymin) / image_height

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Write to .txt
        out_filename = xml_file.replace('.xml', '.txt')
        with open(os.path.join(output_dir, out_filename), 'w') as f:
            f.write('\n'.join(yolo_lines))

if __name__ == "__main__":
    convert_voc_to_yolo(
        xml_dir="./data/VOC_train_upscaled_4x/Annotations",
        output_dir="./data/VOC_train_upscaled_4x_yolo/labels/train",
        img_dir="./data/VOC_train_upscaled_4x/JPEGImages"
    )

