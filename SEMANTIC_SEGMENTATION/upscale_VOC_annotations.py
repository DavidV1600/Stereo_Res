import xml.etree.ElementTree as ET
import os
def upscale_voc_annotation(xml_path, output_path, scale=4):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        # Scale main object box
        bbox = obj.find('bndbox')
        for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
            val = float(bbox.find(tag).text)
            bbox.find(tag).text = str(int(val * scale))

        # Scale parts (if any)
        for part in obj.findall('part'):
            part_box = part.find('bndbox')
            for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                val = float(part_box.find(tag).text)
                part_box.find(tag).text = str(int(val * scale))

    tree.write(output_path)


ann_input_dir = "./data/VOC_train/VOCdevkit/VOC2012/Annotations"
ann_output_dir = "./data/VOC_train_upscaled_4x/Annotations"
os.makedirs(ann_output_dir, exist_ok=True)

for ann_file in os.listdir(ann_input_dir):
    if ann_file.endswith('.xml'):
        upscale_voc_annotation(
            os.path.join(ann_input_dir, ann_file),
            os.path.join(ann_output_dir, ann_file),
            scale=4
        )
