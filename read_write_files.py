import json

def read_json(filename):
    with open(filename, encoding='utf-8') as f:
        obj = json.load(f)
    return obj

def save_as_json_files(objs, filename):
    with open(filename, 'w') as f:
        json.dump(objs, f)

def save_as_txt_files(objs, filename):
    with open(filename, 'w') as f:
        if objs.__class__ == dict:
            for obj in objs:
                f.write(str(obj)+":"+str(objs[obj])+'\n')
        if objs.__class__ == list:
            for obj in objs:
                f.write(str(obj)+'\n')