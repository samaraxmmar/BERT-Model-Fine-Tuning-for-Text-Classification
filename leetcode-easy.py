class Solution:
    def isUgly(self, n):
        if n <= 0:
            return False  # Les nombres négatifs ou 0 ne sont pas des nombres ugly
        
        # Diviser successivement n par 2, 3, ou 5
        for div in [2, 3, 5]:
            while n % div == 0:  # Tant que n est divisible par div
                n //= div  # Diviser n par div
        
        # Si à la fin n est 1, alors c'est un nombre ugly
        return n == 1

# Exemple d'utilisation
solution = Solution()

# Test des exemples donnés
print(solution.isUgly(121))   # True, car 6 = 2 × 3
print(solution.isUgly(1))   # True, car 1 n'a pas de facteurs premiers
print(solution.isUgly(14))  # False, car 14 contient le facteur premier 7
