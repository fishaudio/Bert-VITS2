# N=int(input())
# while N:
#     N-=1
#     n,k=list(map(int,input().split()))
#     nums=list(map(int,input().split())) 
#     sums=[]
#     for i in nums:
#         te=k-(i+k)%k
#         if te==k:
#             sums.append(0)
#         else:
#             sums.append(te)
#     kkk=999999999
#     if k==4 and n>1:
#         ji=len([i for i in nums if i%2==1])
#         if len(nums)-ji>1:kkk=0
#         if len(nums)-ji==1:kkk=1
#         if len(nums)-ji==0 :kkk=2
                    
#     print("dabdjakdskadna ",min(min(sums),kkk))



def test(*p1,**p2):
    print(p1)
    print("---")
    print(p2)
# test(c=3,{"csa":1})    

print({"ca":1}["ca"])