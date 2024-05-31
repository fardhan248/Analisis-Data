        program ftbs
        
        ! deklarasi variabel
        real,allocatable :: x(:), t(:), f(:,:) ! f(nx,nt)
        real :: a , dx, dt, Lt, Lx, pi, lintang, bujur, u
        integer :: nx, nt, ix, it, ierror

!       inisiasi parameter
        pi = 3.14
        lintang = 14*pi/180
        bujur = 5*pi/180
        a = 300 ! m/s (kecepatan/amplitudo adveksi)
        Lt = 24*60*60 ! s (Rentang waktu simulasi)   diubahhh, dari hari ke jam
        Lx = 2*pi*6371000*cos(lintang) ! m (Panjang domain simulasi)
        dx = 6371000*cos(lintang)*bujur ! m (Jarak antar grid)
        dt = 1798 !s (Langkah watu)

!       Hitung array yg dibutuhkan
        nx = int(Lx/dx)+1
        nt = int(Lt/dt)+1 ! (Jumlah langkah waktu)

!       Memasukkan nilai array
        allocate (f(nx,nt))
        allocate (x(nx))
        allocate (t(nt))
        do ix = 1, nx
           x(ix) = ix*dx - dx
        end do 
        do it = 1, nt
           t(it) = it*dt - dt
        end do 
  
!       memasukkan kondisi awal     
!       untuk x = 1 sd x = i-1 bernilai 0
        OPEN(1000, FILE="angin.dat", FORM="UNFORMATTED", ACCESS="STREAM")
        DO i = 1, nx, 1
            READ(1000, IOSTAT = ierror) u
            IF (ierror /= 0) EXIT
            f(i, 1) = u
        END DO
        CLOSE(1000)

!       perhitungan FTBS
        do 101 it = 2, nt ! prediksi untuk langkat waktu t = 2 s.d nt
          ! loop waktu        
          do 102 ix = 1, nx
            ! loop ruang
            if (ix == 1) then
                f(ix, it) = f(ix,it-1)-((a*dt)/dx)*(f(ix,it-1)-f(nx,it-1))
            else
                f(ix,it) = f(ix,it-1)-((a*dt)/dx)*(f(ix,it-1)-f(ix-1,it-1))
            end if
102     continue 
101     continue

!       ========== FORMAT PENULISAN DATA ============
200     format(F15.1,2x,F15.3,2x,F10.3)
!       =============================================


        open(1001,file='hasil.dat', status='unknown') 
                
        do 106 it=1,nt
         do 107 ix=1,nx
           write(1001,200) x(ix),t(it), f(ix,it)
           write(*,200) x(ix),t(it), f(ix,it)
107      continue
106     continue

        write(*,*)
        write(*,*) "Courant number = ", (a*dt)/dx
        write(*,*) "dt = ", dt
        write(*,*) "nt = ", nt

        close(1001)

        end program ftbs
