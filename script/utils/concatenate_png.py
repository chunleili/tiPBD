def concatenate_png(case_name='attach64', imgs=['F1-0', 'F6-0', 'F11-0', 'F16-0', 'F21-0', 'F26-0'], prefix='residuals'):
    from PIL import Image
    import os
    import sys

    if '-case_name' in sys.argv:
        case_name = sys.argv[sys.argv.index('-case_name')+1]

    os.chdir(f"result/{case_name}/png")

    print(os.getcwd())

    image1 = Image.open(f'{prefix}_{imgs[0]}.png')
    image2 = Image.open(f'{prefix}_{imgs[1]}.png')
    image3 = Image.open(f'{prefix}_{imgs[2]}.png')
    image4 = Image.open(f'{prefix}_{imgs[3]}.png')
    image5 = Image.open(f'{prefix}_{imgs[4]}.png')
    image6 = Image.open(f'{prefix}_{imgs[5]}.png')

    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size
    width4, height4 = image4.size
    width5, height5 = image5.size
    width6, height6 = image6.size

    # 计算拼接后的图像大小,两排共6个图像,每排3个
    result_width = max(sum([width1, width2, width3]), sum([width4, width5, width6]))
    result_height = max(height1, height2, height3) + max(height4, height5, height6)

    # 创建一个新的空白图像，大小为拼接后的图像大小
    result_image = Image.new('RGB', (result_width, result_height))

    # 将图像拷贝到结果图像中
    result_image.paste(image1, (0, 0))              # 第一排第一个
    result_image.paste(image2, (width1, 0))         # 第一排第二个
    result_image.paste(image3, (width1+width2, 0))  # 第一排第三个
    result_image.paste(image4, (0, height1))        # 第二排第一个
    result_image.paste(image5, (width4, height2))   # 第二排第二个
    result_image.paste(image6, (width4+width5, height3)) # 第二排第三个

    # 保存拼接后的图像
    result_image.save(f'{case_name}_concat.png')
    result_image.show(f'{case_name}_concat.png')


def concatenate_png3(case_name='attach64', imgs=['F1-0', 'F6-0', 'F11-0']):
    from PIL import Image
    import os
    import sys

    if '-case_name' in sys.argv:
        case_name = sys.argv[sys.argv.index('-case_name')+1]

    os.chdir(f"result/{case_name}/png")

    print(os.getcwd())

    image1 = Image.open(f'{imgs[0]}.png')
    image2 = Image.open(f'{imgs[1]}.png')
    image3 = Image.open(f'{imgs[2]}.png')


    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size


    # 计算拼接后的图像大小,每排3个
    result_width = sum([width1, width2, width3])
    result_height = max(height1, height2, height3)

    # 创建一个新的空白图像，大小为拼接后的图像大小
    result_image = Image.new('RGB', (result_width, result_height))

    # 将图像拷贝到结果图像中
    result_image.paste(image1, (0, 0))              # 第一排第一个
    result_image.paste(image2, (width1, 0))         # 第一排第二个
    result_image.paste(image3, (width1+width2, 0))  # 第一排第三个

    # 保存拼接后的图像
    result_image.save(f'{case_name}_concat.png')
    result_image.show(f'{case_name}_concat.png')



def concatenate_png_variable(case_name='attach64', imgs=['F1-0', 'F6-0', 'F11-0'], nrows=2, ncols=3):
    from PIL import Image
    import os
    import sys

    if '-case_name' in sys.argv:
        case_name = sys.argv[sys.argv.index('-case_name')+1]

    os.chdir(f"result/{case_name}/png")

    print(os.getcwd())

    images = [Image.open(f'{img}.png') for img in imgs]

    widths, heights = zip(*(i.size for i in images))

    # 假设每个图像的大小都相同
    assert all(width == widths[0] for width in widths)
    assert all(height == heights[0] for height in heights)

    result_width = 3*widths[0]
    result_height = 2*heights[0]

    # 创建一个新的空白图像，大小为拼接后的图像大小
    result_image = Image.new('RGB', (result_width, result_height))

    boxes = [(i%ncols, i//ncols) for i in range(nrows*ncols)]

    # 将图像拷贝到结果图像中
    for i, img in enumerate(images):
        result_image.paste(img, (boxes[i][0]*widths[0], boxes[i][1]*heights[0]))

    # 保存拼接后的图像
    result_image.save(f'{case_name}_concat.png')
    result_image.show(f'{case_name}_concat.png')


if __name__ == '__main__':
    concatenate_png()