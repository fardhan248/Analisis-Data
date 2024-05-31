        program ctcs
        
        ! deklarasi variabel
        real,allocatable :: x(:), t(:), f(:,:) ! f(nx,nt)
        real :: a , dx, dt, Lt, Lx, pi
        integer :: nx, nt , xi, xj, i , j , ix, it

!       inisiasi parameter
        a = -300 ! m/s (kecepatan/amplitudo adveksi)
        Lt = 0.5 ! s (Rentang waktu simulasi)
        Lx = 300 ! m (Panjang domain simulasi)
        dx = 5 ! m (Jarak antar grid)
        dt = 0.01666 !s (Langkah waktu)
        pi = 3.14

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

!       Batas fungsi
        xi = 50 ! m
        xj = 110 ! m
        i = int(xi/dx)+1 
        j = int(xj/dx)+1
        
!       memasukkan kondisi awal      
!       untuk x = 1 sd x = i-1 bernilai 0
        f(1:i-1,1) = 0

!       untuk x = i sd x = j bernilai dari suatu fungsi sinus
        do 100 ix = i,j
           f(ix,1) = 100*sin(pi*(x(ix)-50.0)/60.0)
100     continue

!       untuk x = j sd x = nx bernilai 0
        f(j+1:nx,1) = 0
    

!       perhitungan CTCS
        do 101 it = 2, nt-1 ! prediksi untuk langkat waktu t = 2 s.d nt
          ! loop waktu
          ! syarat batas di boundary grid x = 1 dan grid x = nx
          f(1,it) = 0
          f(nx,it) = 0        
          do 102 ix = 2, nx-1
            ! loop ruang
            f(ix,it+1) = f(ix,it-1)-((a*dt)/dx)*(f(ix+1,it-1)-f(ix-1,it-1))
102     continue 
101     continue
 
!       ========== FORMAT PENULISAN DATA ============
200     format(F5.1,2x,F7.3,2x,F20.3)
!       =============================================
201     format("x", 2x, "t", 2x, "u")

        open(1001,file='ctcs3.dat', status='unknown')
         
!        write(1001, 201)
        do 106 it=1,nt
         do 107 ix=1,nx
           write(1001,200) x(ix),t(it), f(ix,it)
           write(*,200) x(ix),t(it), f(ix,it)
107      continue
106     continue

        write(*,*)
        write(*,*) "Courant number = ", (a*dt)/dx

        close(1001)

        end program ctcs
