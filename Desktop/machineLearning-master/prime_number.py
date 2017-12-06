#Description:

#Count the number of prime numbers less than a non-negative number, n.

class Solution():
    def countprimes(self,n):
        if n<=2:
            return 0
        is_primes = [True]*2
        num = 0
        for i in range(2,n):
            if is_primes[i]:
                for j in range(i+1,n,i):
                    is_primes[i] = False
        return num

