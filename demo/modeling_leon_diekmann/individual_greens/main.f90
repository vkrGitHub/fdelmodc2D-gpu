program main

  use mdle_io
  use omp_lib
	
  implicit none
  real :: ft_fx, lt_fx, dfx, ft_fz, lt_fz, dfz

  integer        :: ix, iz, nx, nz, thread_num
  real           :: sz, sx
  character(130) :: f_datum, f_green, f_sel
  character(20)  :: str

  call system("clear")

  !Defines the number of threads to be use
  thread_num = omp_get_max_threads ( )
  write(*,*) '  The number of processors available = ', omp_get_num_procs ( )
  write(*,*) '  The number of threads available    = ', thread_num

  !Input parameters
  call inputdata(ft_fx, lt_fx, dfx, ft_fz, lt_fz, dfz)

  !Number of datums to be deconcatene
  nz = int((lt_fz-ft_fz)/dfz) + 1
  nx = int((lt_fx-ft_fx)/dfx) + 1
  write(*,*) nz, nx

  !$OMP PARALLEL PRIVATE(sx, sz, f_datum, f_green, f_sel)
  !$OMP DO
  do iz = 1,nz
  	!Defines the z-position of the source
  	sz = ft_fz + (iz-1)*dfz

  	do ix = 1,nx
  		!Defines the x-position of the source
  		sx = ft_fx + (ix-1)*dfx

  		!Creates the file name of datum defined
  		write(f_datum, '("./../greens/datums_greens/greens_z",I0,"to",I0,"_x",I0,".0to",I0,".0.su")') int(sz), int(sz), int(ft_fx), int(lt_fx)

  		!Creates the file name of the Green_function
  		write(f_green, '("./files_su/FGreen_z",I0,"_x",I0,".su")') int(sz), int(sx)

  		!Selects the folder correspond to the focal point
  		f_sel = 'suwind < '//trim(f_datum)//' key=fldr min='//trim(str(ix))//' max='//trim(str(ix))//' > '//trim(f_green)
  		call system(f_sel)

  		write(*,*) 'Focal point at (x,z) = (',sx,',',sz,') generated'
  		write(*,*)
  	enddo
  enddo
  !$OMP END DO
  !$OMP END PARALLEL

return
end program main

!Convert an integer to string
character(20) function str(k)

  integer, intent(in) :: k
  write (str, *) k
  str = adjustl(str)
end function str
