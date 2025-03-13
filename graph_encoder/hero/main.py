import argparse
from ruamel.yaml import YAML
import os
from models import HGNN
from models import HGNN_homo
from utils import process

def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    # input arguments
    parser = argparse.ArgumentParser()
    if not custom_key == None:
        custom_key = custom_key.split("+")[0]
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--gpu_num', nargs='?', default='0')
    parser.add_argument('--lr', type = float, default = 0.0005)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--hid_units', type=int, default=256)
    parser.add_argument('--hid_units2', type=int, default=512)
    parser.add_argument('--out_ft', type=int, default=512)
    parser.add_argument('--g_dim', type=int, default=256)
    parser.add_argument('--g_equidim', type=int, default=256)
    parser.add_argument('--p_equidim', type=int, default=256)
    parser.add_argument('--edge_rate', type=float, default=0.9)
    parser.add_argument("--alpha", default=1,
                        help="Reconstruction error coefficient", type=float)
    parser.add_argument("--beta", default=0.1,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument("--gamma", default=1,
                        help="Reconstruction error coefficient", type=float)
    parser.add_argument("--eta", default=1,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument("--lambbda", default=10,
                        help="Independence constraint coefficient", type=float)

    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    args = parser.parse_args()
    return args



def main():
    process.setup_seed(0)
    args = get_args(
        model_name="SHGL",
        dataset="ACM",  #Heterogeneous graph: ACM, Yelp, DBLP, Aminer || Homogeneous graph: photo, computers, cs, physics
        custom_key="Node",  # Node: node classification
    )
    if args.dataset in ["ACM", "Yelp", "DBLP", "Aminer"]:
        embedder = HGNN(args)
    else:
        embedder = HGNN_homo(args)
    embedder.training()

if __name__ == '__main__':
    main()
