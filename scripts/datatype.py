#!/usr/bin/python3
import common
from common import run_experiment
from common import make_minibench_command

if __name__ == '__main__':
	p = common.base_args()
	p = common.common_args(p)
	p.add_argument("--bench_path",
                 type=str,
                 default="../build/examples/datatype",
                 help="Path to the benchmark folder (default: ../build/examples)"
                 )
	p.add_argument("--tiled_sizes",
                 type=int,
                 nargs=2,
                 default=[1,1],
                 help="Tiled layout sizes (default: A=1, B=1)"
                 )
	p.add_argument("--bucket_sizes",
                 type=int,
                 nargs=3,
                 default=[1,1,1],
                 help="Bucket layout sizes (default: A1=1, A2=1, B=1)"
                 )
	p.add_argument("--block_sizes",
                 type=int,
                 nargs=3,
                 default=[1,1,1],
                 help="Block layout sizes (default: A=1, B1=1, B2=1)"
                 )
	p.add_argument("--alternating_sizes",
                 type=int,
                 nargs=4,
                 default=[1,1,1,1],
                 help="Alternating layout sizes (default: A1=1, A2=1, B1=1, B2=1)"
                 )
	args = p.parse_args()
	noop= lambda x: x
	layouts = ["tiled", "block", "bucket", "alternating"]
 
	tmp = args.num_proc
	args.num_proc = 2
	run_experiment(args, "Ping pong: MPI", make_minibench_command(args, "ping_pong/mpi_ping_pong"),noop)
	run_experiment(args, "Ping_pong: EMPI", make_minibench_command(args, "ping_pong/empi_ping_pong"),noop)
	args.num_proc = tmp

	run_experiment(args, "Bidirectional ring: MPI", make_minibench_command(args, "bdring/mpi_bdring"),noop)
	run_experiment(args, "Bidirectional ring: EMPI", make_minibench_command(args, "bdring/empi_bdring"),noop)

	run_experiment(args, "Allreduce: MPI", make_minibench_command(args, "all_reduce/mpi_allreduce"),noop)
	run_experiment(args, "Allreduce: EMPI", make_minibench_command(args, "all_reduce/empi_allreduce"),noop)

	run_experiment(args, "IBcast: MPI", make_minibench_command(args, "ibcast/mpi_ibcast"),noop)
	run_experiment(args, "IBcast: EMPI", make_minibench_command(args, "ibcast/empi_ibcast"),noop)

	run_experiment(args, "Bcast: MPI", make_minibench_command(args, "bcast/mpi_bcast"),noop)
	run_experiment(args, "Bcast: EMPI", make_minibench_command(args, "bcast/empi_bcast"),noop)