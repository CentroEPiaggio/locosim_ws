from nn_trajectory_generator_test_utils import test
from utils.rlg_utils import build_rlg_model
import yaml


def main(args=None):
        pth_path = "/home/jacopo/Documents/locosim_ws/src/rlg_quad_controller/models/Mulinex_2024_02_13-12_00_00/MulinexTerrainNew.pth"
        yaml_path = "/home/jacopo/Documents/locosim_ws/src/rlg_quad_controller/models/Mulinex_2024_02_13-12_00_00/config.yaml"
        with open(yaml_path,"r") as f:
                params = yaml.safe_load(f)
        model = build_rlg_model(pth_path,params)
        test(model)



if __name__ == "__main__":
        main()