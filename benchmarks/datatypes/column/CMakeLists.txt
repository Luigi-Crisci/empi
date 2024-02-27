# # ping pong create_example(column_mpi_ping_pong ping_pong/mpi_ping_pong.cpp)
# create_example(column_empi_ping_pong ping_pong/empi_ping_pong.cpp) # bcast
# create_example(column_mpi_bcast bcast/mpi_bcast.cpp)
# create_example(column_empi_bcast bcast/empi_bcast.cpp) # allgather
# create_example(column_mpi_allgather allgather/mpi_allgather.cpp)
# create_example(column_empi_allgather allgather/empi_allgather.cpp) Send
create_example(column_mpi_send send/mpi_send.cpp)
create_example(column_empi_send send/empi_send.cpp)
create_example(column_pack_send send/pack_send.cpp)

create_example(column_mpi_bcast bcast/mpi_bcast.cpp)
create_example(column_empi_bcast bcast/empi_bcast.cpp)
create_example(column_pack_bcast bcast/pack_bcast.cpp)