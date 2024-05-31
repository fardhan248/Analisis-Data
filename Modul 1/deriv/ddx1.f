      subroutine DDX1 (a,b,l,dx,index)                                         
c**************************************************************************
c  This subroutine performs the second 
c  order finite difference                    
c                                                                              
c  input      : a,l,dx                                                   
c  output     : b               
c  index=1    : forward scheme
c  index=-1   : backward scheme                                                                                
c**************************************************************************    
      real a(l),b(l)    
      integer index
c
      if (index.eq.1) then
		  l1  = l-1                                                                
		  do 2110 i = 1, l1
			 b(i)   =  (a(i+1) - a(i))/(dx)                                               
2110      continue
          b(l)=-9999999
	  end if     
	  
      if (index.eq.-1) then                                       
		  do 2111 i = 2, l
			 b(i)   =  (a(i) - a(i-1))/(dx)                                               
2111      continue
          b(1)=-9999999
      end if                                     
        
c
      return                                                                   
      end
