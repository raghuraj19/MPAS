module mpas_device_utilities
#ifdef MPAS_OPENACC
  use mpi
  use openacc
  use mpas_abort, only : mpas_dmpar_global_abort
  implicit none

  integer :: myrole, mydevice
  integer :: radiation_ranks_per_node, dynamics_ranks_per_node

  contains

  subroutine set_devices_and_roles()
    implicit none
    integer :: world_rank, on_node_comm, local_rank, local_size, ierr
    integer :: socket_rank, increment, limit
    integer :: role_rank, role_size, role_comm
    integer :: numdevices, mydevice
#ifdef SUMMIT_4GPUS
    integer :: indx
    integer, parameter :: device_list(0:3) = (/0,1,3,4/)  ! summit 4-gpu case
#endif
    character(8) :: dynamics_ranks_per_node_str

    call MPI_Comm_rank(MPI_COMM_WORLD, world_rank, ierr)
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, on_node_comm, ierr)
    call MPI_Comm_rank(on_node_comm, local_rank, ierr)
    call MPI_Comm_size(on_node_comm, local_size, ierr)
 
    call get_environment_variable('MPAS_DYNAMICS_RANKS_PER_NODE', value=dynamics_ranks_per_node_str, status=ierr)
    if (ierr /= 0) then
        call mpas_dmpar_global_abort('Error: Could not read environment variable MPAS_DYNAMICS_RANKS_PER_NODE')
    end if
    read(dynamics_ranks_per_node_str, *) dynamics_ranks_per_node
 
    radiation_ranks_per_node = local_size - dynamics_ranks_per_node
 
    ! evenly distribute role assignments over two sockets
    if (local_rank < local_size/2) then
      socket_rank = local_rank
    else
      socket_rank = local_rank - (local_size / 2)
    endif
 
    if (dynamics_ranks_per_node .le. radiation_ranks_per_node) then
      increment = local_size / dynamics_ranks_per_node
      limit = (increment * dynamics_ranks_per_node) / 2 
      if ( mod(socket_rank, increment) == 0 .and. socket_rank < limit ) then
        myrole = 0 ! dynamics  role
      else
        myrole = 1 ! radiation role
      end if
    else ! more radiation ranks than dynamics ranks
      increment = local_size / radiation_ranks_per_node
      limit = (increment * radiation_ranks_per_node) / 2 
      if ( mod(socket_rank, increment) == 0 .and. socket_rank < limit ) then
        myrole = 1 ! radiation role
      else
        myrole = 0 ! dynamics  role
      end if
    end if

    ! assign devices based on rank in the role communicator
    call MPI_Comm_split(on_node_comm, myrole, local_rank, role_comm, ierr)
    call MPI_Comm_rank(role_comm, role_rank, ierr)
    call MPI_Comm_size(role_comm, role_size, ierr)
 
    if (myrole == 0) then
#ifdef SUMMIT_4GPUS
       numdevices = 4 
       indx     = (role_rank * numdevices) / role_size
       mydevice = device_list(indx)
#else
       numdevices = acc_get_num_devices(acc_device_nvidia)
       mydevice = (role_rank * numdevices) / role_size
#endif
       call acc_set_device_num(mydevice, acc_device_nvidia)
    else
       call acc_set_device_type(acc_device_host)
    end if
  end subroutine
#endif
end module
