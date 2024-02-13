def get_img_num_per_cls(data, cls_num, imb_type, imb_factor):
    img_max = len(data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    print(img_num_per_cls)
    print(sum(img_num_per_cls))
    return img_num_per_cls

if __name__ == "__main__":
    data = [0]*10700
    imb_type='exp'
    imb_factor=0.5
    get_img_num_per_cls(data, 2, imb_type,imb_factor)