from pycocotools.coco import COCO
from split_flickr_data import write_data


def process_coco(ann_file):
    data = []
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    print(ann_file, len(imgs))
    for img in imgs:
        filename = img['file_name'].strip()
        ann_ids = coco.getAnnIds(img['id'])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            caption = ann['caption'].strip().replace('\n', '\t')
            data.append(filename+'\t'+caption)
    return data


# def write_data(data, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for d in data:
#             f.write(d+'\n')


if __name__ == '__main__':
    train_file = '/home/a/cv_data/coco2014/annotations/captions_train2014.json'
    val_file = '/home/a/cv_data/coco2014/annotations/captions_val2014.json'

    train_output = '/home/a/MACD/datasets/coco2014/train.tsv'
    val_output = '/home/a/MACD/datasets/coco2014/dev.tsv'

    train_data = process_coco(train_file)
    val_data = process_coco(val_file)

    # write_data(train_data, train_output)
    # write_data(val_data, val_output)

    print(len(train_data), len(val_data))

    write_data(train_data, train_output)
    write_data(val_data, val_output)
