      program DERIV                                                            
c                                                                               
c this simple program computes the first derivative of a function  
c using the second and fourth order accurate schemes. the following
c driver estimates the derivatives of exponential pressure profile:
c    p(z) = po*exp(-a*z)                                         
c
c last edit: Sep 2020
c                                                                               
      parameter(l=10)                                                          
      real z(l),p(l),p1f(l),p1b(l),p2(l),p4(l),anal(l)
c                                       
      data z /0000.,1000.,2000.,3000.,4000.,                                       
     &         5000.,6000.,7000.,8000.,9000./                                    
      data a,p0,dz/0.000125062,1000.,1000./
c
c  initialize the work arrays
c
      do 2100 k = 1, l
         p2(k)  = 0.
         p4(k)  = 0.
2100  continue
c                                  
      do 2102 k = 1, l
c                                                             
c  construct the function p(z)
c                                                 
      p(k)      =  p0*exp(-a*z(k))
c                                                   
c  compute the analytical derivative
c                                           
      anal(k)   = -a*p0*exp(-a*z(k))
c                                             
 2102 continue 
c 
c  compute the first order estimate (forward)                                
c
      call DDX1 (p,p1f,l,dz,1) 
	  
c  compute the first order estimate (backward)                                           
c
      call DDX1 (p,p1b,l,dz,-1) 	  
	  
c  compute the second order estimate                                           
c
      call DDX2 (p,p2,l,dz)                                                   
c
c  compute the fourth order estimate                                           
c
      call DDX4 (p,p4,l,dz)
c
      OPEN(UNIT=6, FILE='data1.dat', STATUS="UNKNOWN") 
c                                                           
      do 2104 k = 1, l                      
		write(6,1001) z(k),p(k), anal(k), p1f(k), p1b(k), p2(k), p4(k)                              
 2104 continue
c                                                                 
1000  format (5x,'p(z)',4x,'dp/dz: (analytical)',4x,'(forward)',7x,'(back&
     &ward)',5x,'(2nd order)',5x,'(4th order)')                                                         
1001  format (f6.1,2x,f8.2,10x,f9.5,4(7x,f9.5))                            
      stop                                                                     
      end                                                                      
