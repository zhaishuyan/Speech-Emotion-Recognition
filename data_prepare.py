import os

pro_path = '/home/admin/dataset/MEAD_pro/'
# pro_path='/Users/dali/Documents/dataset/MEAD_pro'
people = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 'M022', 'M023', 'M024', 'M025', 'M026',
          'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'W011', 'W014', 'W015', 'W016',
          'W018', 'W019']
num=0
for person in people:
    for path, _, files in os.walk(os.path.join(pro_path, person, 'audio')):
        for name in files:
            if name[-4:] == '.wav':
                num+=1
                audio = os.path.join(path, name)
                temp = audio.split('/')
#                 if temp[-2]=='level_3':
                if temp[-3]=='neutral':
                    if not os.path.exists(os.path.join(pro_path, 'audio', temp[-3])):
                        os.makedirs(os.path.join(pro_path, 'audio', temp[-3]))
                    command = 'cp ' + audio + ' ' + os.path.join(pro_path, 'audio', temp[-3],
                                                                 temp[-5] + '-' + temp[-2] + '-' + temp[-1])
                    print(command)
                    os.system(command)
print(num)
