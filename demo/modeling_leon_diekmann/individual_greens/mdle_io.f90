MODULE mdle_io

CONTAINS
	!subroutine inputdata

!===============================================================================
!		     Subroutine to read the input parameters
!===============================================================================
  subroutine inputdata(ft_fx, lt_fx, dfx, ft_fz, lt_fz, dfz)

  implicit none
  real :: ft_fx, lt_fx, dfx, ft_fz, lt_fz, dfz

  character(len=32) :: argfile

  if(1.ne.iargc()) then
    print*,'wrong number of arguments'
    stop
  endif 
  call getarg(1, argfile) 
 
  open(30,file=argfile,status='unknown',action='read',form='formatted')

  read(30,'(t40)')
  read(30,'(t15,f10.4)') ft_fx
  read(30,'(t15,f10.4)') lt_fx
  read(30,'(t15,f10.4)') dfx
  read(30,'(t15,f10.4)') ft_fz
  read(30,'(t15,f10.4)') lt_fz
  read(30,'(t15,f10.4)') dfz

  close(30)

  return
  end subroutine inputdata

END MODULE mdle_io
