import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)

    parser.add_argument("--cuda_id", type=int, default=-1,
                        help="GPU id, -1 means all GPUs visible")
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    dataset_scene = {"replica":["hotel0","office0","office1","office2","office3","office4","room0","room1","room2"]}#
    # dataset_scene = {"replica":["office3"]}
    # scene = args.scene
    cuda_id = args.cuda_id
    target_method = args.method

    algos =["confidence_perspective"] #,"confidence_perspective","confidence_pano"]#
    scenes = dataset_scene[dataset]#
    print(scenes)
    if target_method is not None:
        algos = [target_method]
    print("Detected algorithms:", algos)
    for scene in scenes:
        for algo in algos:
            print(f"\n=== Processing algorithm: {algo} ===")

            cmd = [
                "python", "main.py",
                f"planner={algo}",
                f"scene={dataset}/{scene}",
                "use_gui=False", 
            ]

            print("Running:", " ".join(cmd))


            env = os.environ.copy()
            if cuda_id != -1:
                env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            else:
 
                env.pop("CUDA_VISIBLE_DEVICES", None)

            subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
