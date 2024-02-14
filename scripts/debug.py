class A:
    def func(self):
        pass

    def change(self):
        self.__class__ = ADist

class ADist(A):
    def logprob(self):
        print('ADist logprob')

a = A()
a.change()
a.logprob()
