      subroutine DDX4 (a,b,l,dx)       
c**************************************************************************
c  This subroutine performs the fourth order 
c  finite difference .                          
c                                 
c  input  : a,l,dx                 
c  output : b   
c**************************************************************************
      real a(l),b(l)        
c 
      l1 = l-1                                                                 
      l2 = l-2                                                                 
      l3 = l-3                                                                 
      c1 = 4./3.                                                               
      c2 = 1./3.                                                               
      dx2 = 1./(2.*dx)                                                         
      dx4 = 1./(4.*dx)
      do 2120 i = 3, l2                                                        
          b(i)  = c1*((a(i+1)-a(i-1))*dx2)
     &          - c2*((a(i+2)-a(i-2))*dx4)                    
 2120 continue                                                                 
      b(2)      = -9999999                   
      b(1)      = -9999999                  
      b(l)      = -9999999                                                         
      b(l1)     = -9999999
c                  
      return                                                                  
      end




