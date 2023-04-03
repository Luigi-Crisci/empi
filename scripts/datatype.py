#!/usr/bin/python3
import common
from common import run_experiment
from common import make_datatype_command

if __name__ == '__main__':
    p = common.base_args()
    p = common.common_args(p)
    p.add_argument("--bench_path",
                   type=str,
                   default="../build/examples/datatypes/",
                   help="Path to the benchmark folder (default: ../build/examples/datatypes/)"
                   )
    p.add_argument("--basic",
                   action='store_true',
                   default=False,
                   help="Enable basic datatype benchs"
                   )
    p.add_argument("--struct",
                   action='store_true',
                   default=False,
                   help="Enable struct datatype bench"
                   )
    p.add_argument("--tiled",
                   action='store_true',
                   default=False,
                   help="Enable tiled benchs"
                   )
    p.add_argument("--tiled_sizes",
                   type=int,
                   nargs=2,
                   help="Tiled layout sizes (default: A=1, B=1)"
                   )
    p.add_argument("--bucket",
                   action='store_true',
                   default=False,
                   help="Enable bucket benchs"
                   )
    p.add_argument("--bucket_sizes",
                   type=int,
                   nargs=3,
                   help="Bucket layout sizes (default: A1=1, A2=1, B=1)"
                   )
    p.add_argument("--block",
                   action='store_true',
                   default=False,
                   help="Enable block benchs"
                   )
    p.add_argument("--block_sizes",
                   type=int,
                   nargs=3,
                   help="Block layout sizes (default: A=1, B1=1, B2=1)"
                   )
    p.add_argument("--alternating",
                   action='store_true',
                   default=False,
                   help="Enable alternating benchs"
                   )
    p.add_argument("--alternating_sizes",
                   type=int,
                   nargs=4,
                   help="Alternating layout sizes (default: A1=1, A2=1, B1=1, B2=1)"
                   )
    args = p.parse_args()
    def noop(x): return x

    datatypes = []
    if args.basic:
        datatypes.append("basic")
    if args.struct:
        datatypes.append("struct")
    if(len(datatypes) == 0):
        raise AssertionError("User must provide at least one datatype")
    
    layouts = []
    layout_args = dict()
    if args.tiled and (args.tiled_sizes is not None):
        layouts.append("tiled")
        layout_args["tiled"] = [str(x) for x in args.tiled_sizes]
    if args.bucket and (args.bucket_sizes is not None):
        layouts.append("bucket")
        layout_args["bucket"] = [str(x) for x in args.bucket_sizes]
    if args.block and (args.block_sizes is not None):
        layouts.append("block")
        layout_args["block"] = [str(x) for x in args.block_sizes]
    if args.alternating and (args.alternating_sizes is not None):
        layouts.append("alternating")
        layout_args["alternating"] = [str(x) for x in args.alternating_sizes]

    print(f"Conf: {args}")
    
    for datatype in datatypes:
        print(f"//------------------------- {datatype} datatype ------------------------//")
        for layout in layouts:
            print(f"//--------------- {layout} layout -----------------//")
            tmp = args.num_proc
            args.num_proc = 2
            run_experiment(args, f"{layout} -> Ping pong: MPI", make_datatype_command(args, f"{layout}/{layout}_mpi_ping_pong", layout_args[layout], datatype),noop)
            run_experiment(args, f"{layout} -> Ping_pong: EMPI", make_datatype_command(args, f"{layout}/{layout}_empi_ping_pong", layout_args[layout], datatype),noop)
            args.num_proc = tmp
            
            run_experiment(args, f"{layout} -> Bcast: MPI", make_datatype_command(args, f"{layout}/{layout}_mpi_bcast", layout_args[layout], datatype),noop)
            run_experiment(args, f"{layout} -> Bcast: EMPI", make_datatype_command(args, f"{layout}/{layout}_empi_bcast", layout_args[layout], datatype),noop)
            
            run_experiment(args, f"{layout} -> Allgather: MPI", make_datatype_command(args, f"{layout}/{layout}_mpi_allgather", layout_args[layout], datatype),noop)
            run_experiment(args, f"{layout} -> Allgather: EMPI", make_datatype_command(args, f"{layout}/{layout}_empi_allgather", layout_args[layout], datatype),noop)
            print(f"//---------------------------------------------------//")