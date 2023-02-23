import pdb
import pickle




if __name__ == '__main__':
    f = open('valprep.sh', 'r')
    content = f.read()
    f.close()
    content = content.split('\n')
    labels_content = content[0:1000]
    imgs_content = content[1000:51000]

    labels = []
    for label in labels_content: # 'mkdir -p n13044778'
        labels.append(label.split()[-1])
    labels = sorted(labels)

    labels_dict = {}
    for i, lab in enumerate(labels):
        labels_dict[lab] = i

    imgs = []
    imgs_label = []
    for img in imgs_content:
        imgs.append(img.split()[1])
        imgs_label.append(img.split()[2][0:-1])

    imgs_label_dict = {}
    for i, im in enumerate(imgs):
        # im is like ILSVRC2012_val_00034973.JPEG
        # key is like 00034973
        key = im.split('.')[0].split('_')[2]
        imgs_label_dict[key] = labels_dict[imgs_label[i]]
    
    with open('imagenet_val_labels.pkl', 'wb') as handle:
        pickle.dump({'imgs_label_dict':imgs_label_dict}, handle)
    # pdb.set_trace()

