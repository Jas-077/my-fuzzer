
def get_initial_corpus():
    return ["good"]


def entrypoint(s):
    x = 0
    #print("Bug python file")
    if len(s) > 1:
        #print("Inside if")
        i1 = ord(s[0])
        i2 = ord(s[1])
        for i in range(0, i1 * i2):
            x += 1
    
    if len(s) > 0 and s[0] == 'b':
        #print("Inside b")
        if len(s) > 1 and s[1] == 'a':
            print("Inside a")
            if len(s) > 2 and s[2] == 'd':
                print("Inside d")
                if len(s) > 3 and s[3] == '!':
                    print("Final")
                    exit(219)
                    
                    raise Exception()
