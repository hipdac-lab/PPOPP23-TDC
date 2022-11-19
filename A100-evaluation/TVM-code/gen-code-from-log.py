import codecs
if __name__ == '__main__':
    shape_list = [(64,32,224,224),
                  (64,32,112,112),
                  (32,32,56,56),
                  (64,32,56,56),
                  (64,64,56,56),
                  (32,32,28,28),
                  (64,32,28,28),
                  (96,64,28,28),
                  (160,96,28,28),
                  (192,96,28,28),
                  (32,32,14,14),
                  (64,32,14,14),
                  (128,96,14,14),
                  (192,96,14,14),
                  (32,32,7,7),
                  (64,32,7,7),
                  (96,64,7,7),
                  (192,160,7,7)]
    for shape in shape_list:
        c = shape[0]
        n = shape[1]
        h = shape[2]
        w = shape[3]

        reader = codecs.open("template.template", 'r', 'utf-8')
        lines = reader.readlines()
        template_content = ''
        for line in lines:
            template_content += line
        reader.close()
        source_file = '{}_{}_{}_{}.txt'.format(c, n, h, w)
        reader = codecs.open(source_file, 'r', 'utf-8')
        lines = reader.readlines()
        reader.close()
        source_code = ''
        append = False
        start = 0
        end = 0
        for index, line in enumerate(lines):
            print(line)
            if '__global__ void' in line:
                start = index
            if '******6*******' in line:
                end = index
        for line in lines[start:end]:
            source_code += line
        template_content = template_content.replace('tvm_code_place_holder', source_code + '\n')
        template_content = template_content.replace('#define C place holder', '#define C {}'.format(c))
        template_content = template_content.replace('#define N place holder', '#define N {}'.format(n))
        template_content = template_content.replace('#define H place holder', '#define H {}'.format(h))
        template_content = template_content.replace('#define W place holder', '#define W {}'.format(w))

        grid_block = lines[-1]
        temp_list = grid_block.split('=')
        grid = temp_list[1][0:-8]
        block = temp_list[-1][0:-1]
        #print(grid)
        #print(block)
        #print('********')
        template_content = template_content.replace('dim3_grid_place_holder', 'dim3 grid{};'.format(grid))
        template_content = template_content.replace('dim3_block_place_holder', 'dim3 block{};'.format(block))
        out_file = '{}_{}_{}_{}.cu'.format(c, n, h, w)
        writter = codecs.open(out_file, 'w', 'utf-8')
        writter.write(template_content)

