PROGRAM baca
IMPLICIT NONE
REAL, DIMENSION(73) :: x
REAL :: u
INTEGER :: i, ierror

open(10, file="angin.dat", status="OLD", access="STREAM", form="UNFORMATTED")

do i = 1,73
    read(10, IOSTAT=ierror) u
    if (ierror /= 0) EXIT
    x(i) = u
    !write(*,*) x(i)
end do

write(*,*) x

END PROGRAM baca
