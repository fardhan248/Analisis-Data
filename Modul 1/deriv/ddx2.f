      subroutine DDX2 (a,b,l,dx)                                         
c**************************************************************************
c  This subroutine performs the second 
c  order finite difference                    
c                                                                              
c  input      : a,l,dx                                                   
c  output     : b                                                                                               
c**************************************************************************    
      real a(l),b(l)                                                       
c
      l1  = l-1                                                                
      do 2110 i = 2, l1
         b(i)   =  (a(i+1) - a(i-1))/(2.*dx)                                               
 2110 continue                                            
      b(1) = -9999999                                    
      b(l) = -9999999                                   
c
      return                                                                   
      end





