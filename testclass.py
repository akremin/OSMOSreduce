



class testclass:
    def __init__(self,power):
        self.power = power

    from testclass2 import print_this


if __name__ == '__main__':
    test = testclass(10)
    test.print_this()