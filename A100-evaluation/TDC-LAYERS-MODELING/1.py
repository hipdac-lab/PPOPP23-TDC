import codecs
reader = codecs.open("a2", 'r', 'utf-8')
lines = reader.readlines()
writter = codecs.open("a100-occupancy-unrolled", 'w', 'utf-8')
for line in lines:
    parts = line.split(',')
    #64,32,224,224,1,1,1,1,0,2.96163
    occupancy = float(parts[-3]) / 2
    writter.write('{},{},{},{},{},{},{},{},{},{}'.format(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], occupancy, parts[8], parts[9]))
