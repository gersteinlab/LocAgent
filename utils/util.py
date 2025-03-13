import json

def convert_to_json(obj):
    if isinstance(obj, list):
        res_obj = []
        for o in obj:
            try:
                json_o = json.loads(o.model_dump_json())
            except:
                json_o = o
            res_obj.append(json_o)
        return res_obj
    else:
        try:
            return json.loads(obj.model_dump_json())
        except:
            print(f'{type(obj)} cannot be converted to json directly.')
            return None