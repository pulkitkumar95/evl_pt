import numpy as np
import os
from datetime import datetime
import argparse
import time
import pandas as pd
import socket
import subprocess
import yaml
import wandb

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import getpass

def get_gpu_info(gpu_type=['a6000', 'a5000'], remove_nodes=None, qos='vulc_scav'):
    if qos == 'scav':
        remove_nodes.append('vulcan')
    def run(cmd, print_err=True):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('UTF-8').splitlines()
        except subprocess.CalledProcessError as e:
            # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            if print_err:
                print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            return [cmd.split()[-1]]
    gpudata = run('sinfo -O nodehost,gres -h')
    new_gpu_data = []
    for gpu in gpu_type:
        new_gpu_data += [line.split(' ')[0] for line in gpudata if gpu in line]
    if remove_nodes is not None:
        for node in remove_nodes:
            new_gpu_data = [gpu_node for gpu_node in new_gpu_data if node not in gpu_node]
    assert len(new_gpu_data) > 0, 'No GPU found'
    return ','.join(new_gpu_data)


def get_transfer_commands(data_paths, destination_dir, dataset, touch_file_path):
    transfer_commands = []
    len_check_commands = []
    touch_file_check_command = f'[ ! -f "{touch_file_path}" ]'
    for key, path in data_paths.items():
        if 'split' in key:
            transfer_commands.append(f'cp -r  {path} {destination_dir}')
        else:
            if dataset in {'epickitchens'}:
                if path[-1]!='/':
                    path += '/'
                transfer_commands.append('ln -s {} {}'.format(path, destination_dir))
            else:
                if 'point' in key:

                    condition_check_command = f'[ "${key}" -eq 1 ] || {touch_file_check_command} &&'
                else:
                    condition_check_command = f'[ "${key}" -eq 1 ] &&'
                transfer_commands.append(f'{condition_check_command} {base_sync_command} {path} {destination_dir}')
                folder_name = path.split('/')[-1]
                dest_dir = os.path.join(destination_dir, folder_name)
                len_check_commands.append(base_len_check_command.format(path, dest_dir, key))
                print(key)
    return transfer_commands, len_check_commands
    
    

qos_dict = {"sailon" : {"nhrs" : 2, "cores": 16, "mem":128},
            "scav" : {"nhrs" : 72, "cores": 92, "mem":500},
            "vulc_scav" : {"nhrs" : 72, "cores": 32, "mem":220},
            "vulc_exe" : {"nhrs" : 24*7, "cores": 32, "mem":220},

            "zara" : {"nhrs" : 72, "cores": 92, "mem":500},
            
            "cml_scav" : {"nhrs" : 72, "cores": 16, "mem":128}, 

            "high" : {"gpu":4, "cores": 16, "mem":200, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168},
            "tron" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}




def check_qos(args):
    
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=72)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='out_new_pt')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--qos', default="vulc_scav", type=str, help='Qos to run')
parser.add_argument('--gpu', default=8, type=int, help='Number of gpus')
parser.add_argument('--cores', default=32, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=220, type=int, help='RAM in G')
parser.add_argument('--exp_name', required=True, type=str, help='Experiment name')
parser.add_argument('--batch_size', default=64, type=int, help='Experiment name')
parser.add_argument('--test_batch_size', default=32, type=int, help='Experiment name')

parser.add_argument('--split', type=str, default='compositional',  help='Experiment name')

parser.add_argument('--dataset', type=str, default='k400',  help='Experiment name')

parser.add_argument('--num_frames', type=int, default=16,  help='Number of frames to be used')
parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate to use')



parser.add_argument('--touch_file_name', type=str, default='_all_ssv2.txt',  help='Experiment name')
parser.add_argument('--gpu_to_use', type=str, default='h100',  help='Experiment name')





parser.add_argument('--use_points', action='store_true')

parser.add_argument('--task', type=str, default='classification',  help='GPU to use')
parser.add_argument('--max_epoch', type=int, default=35,  help='Number of points to sample')
parser.add_argument('--use_docker', action='store_true')
parser.add_argument('--seed', type=int, default=42,  help='Random seed')
parser.add_argument('--point_grid_size', type=int, default=16,  help='Point grid size')

parser.add_argument('--model_type', type=str, default='evlbasic',  help='model type')







# parser.add_argument('--batch_size', default=64, type=int, help='Batch size')





# parser.add_argument('--path', default='/fs/vulcan-projects/actionbytes/vis/ab_training_run3_rerun_32_0.0001_4334_new_dl_nocasl_checkpoint_best_dmap_ab_info.hkl')
# parser.add_argument('--num_ab', default= 100000, type=int, help='number of actionbytes')

args = parser.parse_args()

if 'nexus' in socket.gethostname():
    gpu_types = ['a6000','a5000', 'a4000']
else:
    gpu_types = ['h100', 'a100']
remove_nodes = ['cml17', 'cml20', 'cml28', 'clip', 'gamma']

if 'nexus' in socket.gethostname():
    nodes = get_gpu_info(gpu_types,remove_nodes, qos=args.qos)
else:
    nodes = ''

 

args = parser.parse_args()
time_str = str(int(time.time()))

env = f'{args.dataset}/{args.exp_name}/{args.split}/{time_str}'
setattr(args, 'env', env)
setattr(args, 'output_dir', args.output_dirname)

output_dir = os.path.join(args.base_dir, args.output_dirname, env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



print("Output Directory: %s" % output_dir)
step = 1


params = [(i,) for i in range(1)]



dataset = args.dataset
print("Running on dataset: %s" % dataset) 

point_info_name = 'cotracker2_16_uniform_8_corrected'




print(len(params))
print('Running with Number of frames: %d' % args.num_frames)
print('Running with lr: %f' % args.lr)
       
temporal_skip = None
hostname = socket.gethostname()
if 'nexus' in hostname:
    hostname = 'nexus'
    base_sync_command = f'~/msrsync/msrsync3 -p {args.cores} -P'
elif 'zara' in hostname:
    hostname = 'zaratan'
    base_sync_command = f'~/scratch/msrsync/msrsync3 -p {args.cores} -P'

else:
    raise ValueError(f'{hostname} not found in paths.yaml')
condition_check_command = '[ "${}" -eq 1 ] &&'
path_file = 'paths.yaml'
data_paths_dict = yaml.safe_load(open(path_file, 'r'))


if dataset in data_paths_dict:
    data_paths = data_paths_dict[dataset][hostname]
    destination_dir = data_paths['destination']
    split_info_dir = data_paths['split_info']
    del data_paths['destination']
    del data_paths['split_info']
else:
    raise ValueError('Dataset not found in paths.yaml')


# for few shot, everything is being transferred to scratch1
destination_dir = destination_dir.replace('scratch0', '$SCRATCH_DIR').replace('scratch1', '$SCRATCH_DIR')
if 'nexus' in hostname:
    user_running = os.getlogin()
    destination_dir = destination_dir.replace('pulkit',user_running )

touch_file_path = os.path.join(destination_dir, point_info_name, args.touch_file_name)
if args.use_points:
    data_paths['points_info'] = os.path.join(data_paths['points_info'], point_info_name)
else:
    del data_paths['points_info']
transfer_commands = []
base_len_check_command = "source check_num_files.sh {} {} {}"
len_check_commands = []

if args.dataset == 'k400':
    num_classes = 400
elif args.dataset == 'ssv2':
    num_classes = 174
else:
    raise ValueError(f'{args.dataset} not supported')




port_start = 1000 +  (int(np.random.uniform()*10e5) % 64000)
output_dir = f'{args.base_dir}/{args.output_dir}/{args.env}/'

use_docker = False
if 'zara' in hostname:
    transfer_commands = []
    use_docker = True
    docker_command = 'singularity'
    #bind_paths = '/scratch/zt1/project/abhinav2-prj/user/pulkit'
    username = getpass.getuser()
    bind_paths = '/scratch/zt1/project/abhinav2-prj/user/{}'.format(username)

    image_path = '/scratch/zt1/project/abhinav2-prj/user/pulkit/orvit_pt/cotracker.sif'
    #path_to_transfer = os.path.join('/tmp/pulkit/', args.dataset)

    path_to_transfer = os.path.join('/tmp/{}/'.format(username), args.dataset)
    
    
    if args.dataset == 'ssv2':
        rsync_command = f'~/msrsync/msrsync3 -p {args.cores} '
    else:
        rsync_command = 'rsync -a'
    full_docker_command = f'{docker_command} exec -B {bind_paths} --nv {image_path} '

    if args.dataset == 'k400':
        for split in ['train', 'val']:
            vid_data_dir = data_paths[f'{split}_videos']
            vid_transfer_command = f'{full_docker_command} {rsync_command} {vid_data_dir} {path_to_transfer}'
            transfer_commands.append(vid_transfer_command)
    else:
        vid_data_dir = data_paths['videos']
        vid_transfer_command = f'{rsync_command} {vid_data_dir} {path_to_transfer}'
        transfer_commands.append(vid_transfer_command)
    
    # split_info_transfer_command = f'{full_docker_command} rsync -a {split_info_dir} {path_to_transfer}'
    # transfer_commands.append(split_info_transfer_command)

    if args.use_points:
        pt_data_dir = data_paths['points_info']
        pt_transfer_command = f'{full_docker_command} {rsync_command} {pt_data_dir} {path_to_transfer}'
        transfer_commands.append(pt_transfer_command)
    destination_dir = path_to_transfer

else:
    if dataset !='k400':
        transfer_commands, len_check_commands = get_transfer_commands(data_paths, 
                                                    destination_dir, dataset,
                                                    touch_file_path)
    else:
        destination_dir = data_paths['videos']
# breakpoint()


with open(f'{args.base_dir}/{args.output_dir}/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/{args.output_dir}/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/{args.output_dir}/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/{args.output_dir}/{args.env}/name.txt', "w") as namefile:

    for i, (checkk) in enumerate(params):
        log_output_dir = os.path.join(output_dir, f'test_{i}')
        os.makedirs(log_output_dir, exist_ok=True)
        now = datetime.now()
        master_port = port_start + i
        wandb_id = wandb.util.generate_id()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = f'test_{i}'
        cmd = ''
        if use_docker:
            cmd = f'{docker_command} exec -B {bind_paths} --nv {image_path} '
        cmd += f'torchrun --nproc_per_node={args.gpu} --master_port={master_port} '
        cmd += f'main.py --exp_name {args.exp_name} '
        cmd += '--num_steps 50000 '
        cmd += '--backbone ViT-B/16-lnpre '
        cmd += '--backbone_type clip '
        cmd += '--backbone_path checkpoints/ViT-B-16.pt '
        cmd += '--decoder_num_layers 4 '
        cmd += '--decoder_qkv_dim 768 '
        cmd += '--decoder_num_heads 12 '
        cmd += f'--num_classes {num_classes} '
        cmd += f'--checkpoint_dir {log_output_dir} '
        cmd += '--auto_resume '
        cmd += '--train_list_path ' + os.path.join(split_info_dir, args.dataset, 'train.txt') + ' '
        cmd += '--val_list_path ' + os.path.join(split_info_dir, args.dataset, 'val.txt') + ' '
        cmd += f'--batch_size {args.batch_size} '
        cmd += '--batch_split 1 '
        cmd += '--auto_augment rand-m7-n4-mstd0.5-inc1 '
        cmd += '--mean 0.48145466 0.4578275 0.40821073 '
        cmd += '--std 0.26862954 0.26130258 0.27577711 '
        cmd += f'--num_workers {args.cores} '
        cmd += f'--num_frames {args.num_frames} '
        cmd += '--sampling_rate 16 '
        cmd += '--num_spatial_views 3 '
        cmd += '--num_temporal_views 1 '
        cmd += f'--vid_base_dir {destination_dir} '
        cmd += f'--test_batch_size {args.test_batch_size} '
        cmd += f'--wandb_id {wandb_id} '
        cmd += f'--dataset {args.dataset} '
        cmd += f'--model_type {args.model_type} '
            
       
        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')
        #break
###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'{args.exp_name}.slurm')
slurm_command = "sbatch %s" % slurm_script_path



# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{len(params)}\n")
    #slurmfile.write(f"#SBATCH --array=1-10\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    slurmfile.write("#SBATCH --nodes=1\n")
    # slurmfile.write("#SBATCH --exclude=vulcan[00-23]\n")

    
    args = check_qos(args)

    
    if "scav" in args.qos or "tron" in args.qos or "zara" in args.qos or "vulc_exe" in args.qos:
        if 'nexus' in hostname:
            if args.qos == "scav":
                slurmfile.write("#SBATCH --account=scavenger\n")
                slurmfile.write("#SBATCH --qos scavenger\n")
                slurmfile.write("#SBATCH --partition scavenger\n")
            

            elif args.qos == "vulc_scav":
                slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
                slurmfile.write("#SBATCH --qos vulcan-scavenger\n")
                slurmfile.write("#SBATCH --partition vulcan-scavenger\n")
            elif args.qos == 'cml_scav':
                slurmfile.write("#SBATCH --account=cml-scavenger\n")
                slurmfile.write("#SBATCH --qos cml-scavenger\n")
                slurmfile.write("#SBATCH --partition cml-scavenger\n")
            elif args.qos == 'vulc_exe':
                slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
                slurmfile.write("#SBATCH --qos vulcan-exempt\n")
                slurmfile.write("#SBATCH --partition vulcan-ampere\n")
        else:
            slurmfile.write("#SBATCH --account=abhinav2-prj-cmsc\n")
            if args.qos == "scav":
                slurmfile.write("#SBATCH --qos scavenger\n")
                slurmfile.write("#SBATCH --partition scavenger\n")
            else:
                slurmfile.write("#SBATCH --partition gpu\n")



    
            
        
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

        if not args.gpu is None:
            # if hostname in {'nexus', 'vulcan'}:
            if 'nexus' in hostname:
                #slurmfile.write(f'#SBATCH --gres=gpu:{args.gpu}\n')

                #TODO debug
                slurmfile.write(f'#SBATCH --gres=gpu:rtxa5000:{args.gpu}\n')
            else:

                if args.gpu_to_use == 'a100':
                    slurmfile.write(f'#SBATCH --gres=gpu:a100:{args.gpu}\n')
                else:
                    slurmfile.write(f'#SBATCH --gres=gpu:h100:{args.gpu}\n')
                    
        else:
            raise ValueError("Specify the gpus for scavenger")
    elif args.qos == 'medium':
        # slurmfile.write("#SBATCH  --qos vulcan-high\n")
        # slurmfile.write("#SBATCH --partition vulcan-ampere\n")
        # slurmfile.write("#SBATCH --account=vulcan-abhinav\n")


        slurmfile.write("#SBATCH  --qos medium\n")
        slurmfile.write("#SBATCH --partition tron\n")

        slurmfile.write("#SBATCH --account=nexus\n")
    
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        # slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)

        #TODO debug
        slurmfile.write("#SBATCH --gres=gpu:rtxa6000:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
    if 'nexus' in hostname:
        slurmfile.write("#SBATCH --nodelist=%s\n" % nodes)
    
    
    
    slurmfile.write("\n")
    #slurmfile.write("export MKL_SERVICE_FORCE_INTEL=1\n")p
    slurmfile.write("cd " + os.getcwd() + '\n')
    slurmfile.write("module load ffmpeg\n")
    slurmfile.write("export MKL_THREADING_LAYER=GNU\n")
    slurmfile.write("export SCRATCH_DIR\n")
    if 'nexus' in hostname:
        slurmfile.write("source /fs/cfar-projects/actionloc/new_miniconda/bin/activate\n")
        slurmfile.write("conda activate pips2\n")
        slurmfile.write('[ -d "/scratch1" ] && SCRATCH_DIR="scratch1" || SCRATCH_DIR="scratch0"\n')
        hub_home = '/fs/cfar-projects/actionloc'
    else:
        slurmfile.write("export SCRATCH_DIR=tmp\n")
        slurmfile.write("export WANDB_MODE=offline\n")

        #hub_home = '/scratch/zt1/project/abhinav2-prj/user/pulkit/'

        username = getpass.getuser()
        hub_home = '/scratch/zt1/project/abhinav2-prj/user/{}/'.format(username)


    slurmfile.write(f'export TORCH_HOME={hub_home}\n')
    slurmfile.write(f'export HF_HOME={hub_home}huggingface\n')

    slurmfile.write(f'mkdir -p {destination_dir}\n')
    if 'zara' in hostname:
        
        slurmfile.write("module load singularity\n")


    for len_check_command in len_check_commands:
        slurmfile.write(f'{len_check_command}\n')
    for transfer_command in transfer_commands:
        slurmfile.write(f'{transfer_command}\n')

    
    # slurmfile.write("cd ./libs/utils\n")
    # slurmfile.write("python setup.py install --user\n")
    # slurmfile.write("cd ../..\n")

    slurmfile.write(f'touch {touch_file_path}\n')

    
    slurmfile.write(f"srun --export=ALL,SCRATCH_DIR=$SCRATCH_DIR --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dir}/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dir}/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/{args.output_dir}/{args.env}/now.txt | tail -n 1)\n")
    slurmfile.write("\n")
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour {}".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs, port_start))
print(point_info_name)
if not args.dryrun:
   os.system("%s &" % slurm_command)
else:
   #set env variable to the command
   os.system("export run='%s'" % slurm_command)
